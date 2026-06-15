#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Runner;
using OpenCVForUnity.UnityIntegration.Worker.DnnModule;
using PaddleOCRWithOpenCVForUnity;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace PaddleOCRWithOpenCVForUnityExample
{
    /// <summary>
    /// PaddleOCR example for a handwriting canvas. Optionally runs inference at intervals while writing;
    /// runs one inference after stroke end, Clear, or inference settings change.
    /// </summary>
    /// <remarks>
    /// Add <see cref="PaddleOCRComponent"/> to the same GameObject and configure model paths.
    /// While writing, when <see cref="SubmitOCRWhileWriting"/> is true,
    /// attempts <see cref="PaddleOCRComponent.Submit(OpenCVForUnity.CoreModule.Mat)"/> at <see cref="WhileWritingOCRIntervalSeconds"/> intervals.
    /// Single inference is deferred while <see cref="PaddleOCRComponent.IsWorkInFlight"/> and retried after completion.
    /// Assign <see cref="OnHandwritingStrokeEnded"/> to <see cref="HandwritingCanvas"/> On Stroke Ended and
    /// <see cref="OnInferenceWorkCompletedWithParsedResult"/> to <see cref="PaddleOCRComponent.WorkCompletedWithParsedResult"/> in the Inspector.
    /// </remarks>
    [RequireComponent(typeof(PaddleOCRComponent))]
    public class HandwritingOCRExample : MonoBehaviour
    {
        // Public Fields
        [Header("PaddleOCR")]
        [Tooltip("PaddleOCR inference component (fetched from the same GameObject when unset).")]
        public PaddleOCRComponent PaddleOCR;

        [Header("Input")]
        [Tooltip("Handwriting area in the scene (RawImage + TextureSelector + HandwritingCanvas).")]
        public HandwritingCanvas Canvas;

        [Header("Output")]
        [Tooltip("TMP InputField that displays recognition results.")]
        public TMP_InputField RecognitionResultField;

        [Header("Line thickness")]
        [Tooltip("Line thickness slider (1–50).")]
        public Slider LineThicknessSlider;

        [Tooltip("TMP InputField for numeric line thickness input.")]
        public TMP_InputField LineThicknessInputField;

        [Header("Line color")]
        [Tooltip("Line color ToggleGroup.")]
        public ToggleGroup LineColorToggleGroup;

        [Header("OCR timing")]
        [Tooltip("Toggle for interval inference while writing. Inspector: SubmitOCRWhileWritingToggle → OnSubmitOCRWhileWritingToggleValueChanged.")]
        public Toggle SubmitOCRWhileWritingToggle;

        [Tooltip("When true, attempts Submit at WhileWritingOCRIntervalSeconds intervals while writing (pointer pressed).")]
        public bool SubmitOCRWhileWriting;

        [Tooltip("Minimum interval (seconds) for Submit while writing. Used only when SubmitOCRWhileWriting is true.")]
        public float WhileWritingOCRIntervalSeconds = 0.5f;

        [Header("FpsMonitor")]
        [Tooltip("Display target for inference info. Fetched from the same GameObject when unset.")]
        public FpsMonitor FpsMonitor;

        // Private Fields
        private Mat _bgrMat;
        private float _lastOCRSubmitUnscaledTime = float.NegativeInfinity;
        private bool _submitOncePending;
        private bool _syncingLineThicknessUi;
        private bool _syncingLineColorUi;

        // Unity Lifecycle Methods

        /// <summary>
        /// On startup, waits for <see cref="PaddleOCRComponent"/> auto-initialization to complete,
        /// then sets up the BGR <see cref="Mat"/> and line thickness, color, and OCR timing UI.
        /// </summary>
        private async void Start()
        {
            OpenCVDebug.SetDebugMode(true);

            if (PaddleOCR == null)
            {
                Debug.LogError($"{nameof(HandwritingOCRExample)}: {nameof(PaddleOCRComponent)} is missing.");
                return;
            }

            // AutoInitializeOnStart … InitializeInferenceAsync starts automatically in PaddleOCRComponent.Awake (Inspector default).
            await PaddleOCR.WaitForInitializationAsync();
            if (!PaddleOCR.IsInitialized)
            {
                Debug.LogWarning($"{nameof(HandwritingOCRExample)}: {nameof(PaddleOCRComponent)} is not initialized.");
                return;
            }

            UpdateFpsMonitorInferenceInfo();

            _bgrMat = new Mat(HandwritingCanvas.CANVAS_SIZE, HandwritingCanvas.CANVAS_SIZE, CvType.CV_8UC3);
            SyncLineThicknessUiFromCanvas();
            SyncLineColorUiFromCanvas();
            SyncSubmitOCRWhileWritingToggleFromField();
        }

        /// <summary>
        /// Each frame, attempts pending single inference and submits OCR at the configured interval while writing.
        /// </summary>
        private void Update()
        {
            if (Canvas == null || PaddleOCR == null || !PaddleOCR.IsInitialized)
                return;

            TrySubmitPendingOnceOCR();

            if (!Canvas.IsPointerDown)
                return;

            if (!SubmitOCRWhileWriting)
                return;

            if (Time.unscaledTime - _lastOCRSubmitUnscaledTime < WhileWritingOCRIntervalSeconds)
                return;

            if (TrySubmitCanvasMat())
                _lastOCRSubmitUnscaledTime = Time.unscaledTime;
        }

        /// <summary>
        /// Disposes the BGR <see cref="Mat"/> and disables OpenCV debug mode.
        /// </summary>
        private void OnDestroy()
        {
            _bgrMat?.Dispose(); _bgrMat = null;
            OpenCVDebug.SetDebugMode(false);
        }

        // Public Methods

        /// <summary>
        /// Returns to the main menu. Wire from the Back button in the Inspector.
        /// </summary>
        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("PaddleOCRWithOpenCVForUnityExample");
        }

        /// <summary>
        /// Clears the handwriting canvas and schedules one OCR run on the cleared content.
        /// Wire from the Clear button in the Inspector.
        /// </summary>
        public void OnClearButtonClick()
        {
            if (Canvas != null)
                Canvas.ClearCanvas();

            _submitOncePending = true;
        }

        /// <summary>
        /// Toggles interval inference while writing (Inspector: SubmitOCRWhileWritingToggle → this method).
        /// </summary>
        /// <param name="isOn">True to enable interval inference.</param>
        public void OnSubmitOCRWhileWritingToggleValueChanged(bool isOn)
        {
            SubmitOCRWhileWriting = isOn;
        }

        /// <summary>
        /// Changes line thickness from the slider (Inspector: LineThicknessSlider → this method).
        /// </summary>
        /// <param name="value">Slider value (1–50).</param>
        public void OnLineThicknessSliderValueChanged(float value)
        {
            if (_syncingLineThicknessUi)
                return;

            ApplyLineThickness(Mathf.RoundToInt(value));
        }

        /// <summary>
        /// Changes line thickness from numeric input (Inspector: LineThicknessInputField → this method).
        /// </summary>
        /// <param name="text">Entered thickness (integer). On parse failure, syncs UI to canvas state.</param>
        public void OnLineThicknessInputFieldEndEdit(string text)
        {
            if (_syncingLineThicknessUi)
                return;

            if (int.TryParse(text, out int parsed))
            {
                ApplyLineThickness(parsed);
                return;
            }

            SyncLineThicknessUiFromCanvas();
        }

        /// <summary>
        /// Sets stroke color to dark gray (Inspector: DarkGrayColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnDarkGrayColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.DarkGray, isOn);
        }

        /// <summary>
        /// Sets stroke color to red (Inspector: RedColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnRedColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.Red, isOn);
        }

        /// <summary>
        /// Sets stroke color to green (Inspector: GreenColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnGreenColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.Green, isOn);
        }

        /// <summary>
        /// Sets stroke color to blue (Inspector: BlueColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnBlueColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.Blue, isOn);
        }

        /// <summary>
        /// Sets stroke color to orange (Inspector: OrangeColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnOrangeColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.Orange, isOn);
        }

        /// <summary>
        /// Sets stroke color to purple (Inspector: PurpleColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnPurpleColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.Purple, isOn);
        }

        /// <summary>
        /// Sets stroke color to black (Inspector: BlackColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnBlackColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.Black, isOn);
        }

        /// <summary>
        /// Sets stroke color to white (eraser) (Inspector: WhiteColorToggle → this method).
        /// </summary>
        /// <param name="isOn">Applies color only when the Toggle turns ON.</param>
        public void OnWhiteColorToggleValueChanged(bool isOn)
        {
            OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind.White, isOn);
        }

        /// <summary>
        /// Called after inference UI settings change (Inspector: InferenceSettingsChanged → this method).
        /// Updates FPS display and runs one Mat inference on the current canvas content.
        /// </summary>
        public void OnPaddleOCRInferenceSettingsChanged()
        {
            UpdateFpsMonitorInferenceInfo();
            _submitOncePending = true;
        }

        /// <summary>
        /// Schedules one OCR run on stroke end (Inspector: On Stroke Ended → this method).
        /// </summary>
        public void OnHandwritingStrokeEnded()
        {
            _submitOncePending = true;
        }

        /// <summary>
        /// Called from <see cref="PaddleOCRComponent.WorkCompletedWithParsedResult"/> (Inspector: Work Completed With Parsed Result → this method).
        /// On success, updates <see cref="RecognitionResultField"/> and retries pending single inference if any.
        /// </summary>
        /// <param name="result">OCR completion result (kind, recognized text, scores).</param>
        public void OnInferenceWorkCompletedWithParsedResult(PaddleOCRParsedResult result)
        {
            if (result.Kind != WorkCompletionKind.Succeeded)
            {
                if (result.Kind == WorkCompletionKind.Faulted && !string.IsNullOrEmpty(result.ErrorMessage))
                    Debug.LogWarning($"{nameof(HandwritingOCRExample)} OCR: {result.ErrorMessage}");
                TrySubmitPendingOnceOCR();
                return;
            }

            List<(string text, float score)> recognitions = result.Recognitions;
            string recognitionText = recognitions == null || recognitions.Count == 0
                ? string.Empty
                : string.Join("\n", recognitions.Select(r => $"{r.text} ({r.score:F4})"));

            if (RecognitionResultField != null)
            {
                RecognitionResultField.text = recognitionText;
                RecognitionResultField.stringPosition = 0;
                RecognitionResultField.caretPosition = 0;
                if (RecognitionResultField.verticalScrollbar != null)
                    RecognitionResultField.verticalScrollbar.value = 0;
            }

            Debug.Log(recognitionText);

            TrySubmitPendingOnceOCR();
        }

        // Private Methods

        /// <summary>
        /// Submits the pending canvas once via <see cref="PaddleOCRComponent.Submit(OpenCVForUnity.CoreModule.Mat)"/> after stroke end, Clear, etc.
        /// Retries next frame or after inference completes when <see cref="PaddleOCRComponent.IsWorkInFlight"/>.
        /// For synchronous inference, the completion event fires inside <see cref="PaddleOCRComponent.Submit(OpenCVForUnity.CoreModule.Mat)"/>;
        /// clears the pending flag before submit to prevent re-entrant consecutive Submits.
        /// </summary>
        private void TrySubmitPendingOnceOCR()
        {
            if (!_submitOncePending)
                return;

            if (PaddleOCR.IsWorkInFlight)
                return;

            _submitOncePending = false;

            if (!TrySubmitCanvasMat())
            {
                _submitOncePending = true;
                return;
            }

            _lastOCRSubmitUnscaledTime = Time.unscaledTime;
        }

        /// <summary>
        /// Converts the <see cref="HandwritingCanvas"/> RGBA <see cref="Mat"/> to BGR and submits OCR.
        /// </summary>
        /// <returns>True if Submit succeeded.</returns>
        private bool TrySubmitCanvasMat()
        {
            Mat rgba = Canvas.CanvasMat;
            if (rgba == null)
                return false;

            Imgproc.cvtColor(rgba, _bgrMat, Imgproc.COLOR_RGBA2BGR);
            return PaddleOCR.Submit(_bgrMat);
        }

        /// <summary>
        /// Applies line thickness to the canvas and syncs the slider and InputField.
        /// </summary>
        /// <param name="thickness">Thickness to apply (expected to be clamped).</param>
        private void ApplyLineThickness(int thickness)
        {
            if (Canvas != null)
                Canvas.LineThickness = Mathf.Clamp(thickness, HandwritingCanvas.MIN_LINE_THICKNESS, HandwritingCanvas.MAX_LINE_THICKNESS);

            SyncLineThicknessUiFromCanvas();
        }

        /// <summary>
        /// Reflects <see cref="SubmitOCRWhileWriting"/> on the Toggle UI (without firing events).
        /// </summary>
        private void SyncSubmitOCRWhileWritingToggleFromField()
        {
            if (SubmitOCRWhileWritingToggle != null)
                SubmitOCRWhileWritingToggle.SetIsOnWithoutNotify(SubmitOCRWhileWriting);
        }

        /// <summary>
        /// Reflects canvas line thickness on the slider and InputField (without firing events).
        /// </summary>
        private void SyncLineThicknessUiFromCanvas()
        {
            int thickness = Canvas != null
                ? Canvas.LineThickness
                : HandwritingCanvas.MIN_LINE_THICKNESS;

            _syncingLineThicknessUi = true;

            if (LineThicknessSlider != null)
            {
                LineThicknessSlider.minValue = HandwritingCanvas.MIN_LINE_THICKNESS;
                LineThicknessSlider.maxValue = HandwritingCanvas.MAX_LINE_THICKNESS;
                LineThicknessSlider.wholeNumbers = true;
                LineThicknessSlider.SetValueWithoutNotify(thickness);
            }

            if (LineThicknessInputField != null)
                LineThicknessInputField.SetTextWithoutNotify(thickness.ToString());

            _syncingLineThicknessUi = false;
        }

        /// <summary>
        /// When a color Toggle turns ON, applies the corresponding stroke color.
        /// </summary>
        /// <param name="colorKind">Color to apply.</param>
        /// <param name="isOn">Whether the Toggle is ON.</param>
        private void OnLineColorToggleValueChanged(HandwritingCanvas.StrokeColorKind colorKind, bool isOn)
        {
            if (_syncingLineColorUi || !isOn)
                return;

            ApplyStrokeColor(colorKind);
        }

        /// <summary>
        /// Applies stroke color to the canvas and syncs <see cref="LineColorToggleGroup"/>.
        /// </summary>
        /// <param name="colorKind">Color to apply.</param>
        private void ApplyStrokeColor(HandwritingCanvas.StrokeColorKind colorKind)
        {
            if (Canvas != null)
                Canvas.StrokeColor = colorKind;

            SyncLineColorUiFromCanvas();
        }

        /// <summary>
        /// Syncs Toggles in <see cref="LineColorToggleGroup"/> to match the canvas stroke color.
        /// </summary>
        private void SyncLineColorUiFromCanvas()
        {
            if (LineColorToggleGroup == null || Canvas == null)
                return;

            Toggle activeToggle = GetLineColorToggle(Canvas.StrokeColor);
            if (activeToggle == null)
                return;

            _syncingLineColorUi = true;

            foreach (Toggle toggle in LineColorToggleGroup.GetComponentsInChildren<Toggle>(true))
                toggle.SetIsOnWithoutNotify(toggle == activeToggle);

            _syncingLineColorUi = false;
        }

        /// <summary>
        /// Finds the Toggle for a stroke color by name in <see cref="LineColorToggleGroup"/>.
        /// </summary>
        /// <param name="colorKind">Color to look up.</param>
        /// <returns>The Toggle if found; otherwise null.</returns>
        private Toggle GetLineColorToggle(HandwritingCanvas.StrokeColorKind colorKind)
        {
            if (LineColorToggleGroup == null)
                return null;

            foreach (Toggle toggle in LineColorToggleGroup.GetComponentsInChildren<Toggle>(true))
            {
                switch (colorKind)
                {
                    case HandwritingCanvas.StrokeColorKind.DarkGray when toggle.name == "DarkGrayColorToggle":
                    case HandwritingCanvas.StrokeColorKind.Red when toggle.name == "RedColorToggle":
                    case HandwritingCanvas.StrokeColorKind.Green when toggle.name == "GreenColorToggle":
                    case HandwritingCanvas.StrokeColorKind.Blue when toggle.name == "BlueColorToggle":
                    case HandwritingCanvas.StrokeColorKind.Orange when toggle.name == "OrangeColorToggle":
                    case HandwritingCanvas.StrokeColorKind.Purple when toggle.name == "PurpleColorToggle":
                    case HandwritingCanvas.StrokeColorKind.Black when toggle.name == "BlackColorToggle":
                    case HandwritingCanvas.StrokeColorKind.White when toggle.name == "WhiteColorToggle":
                        return toggle;
                }
            }

            return null;
        }

        /// <summary>
        /// Displays DNN backend, target, and async inference settings on <see cref="FpsMonitor"/>.
        /// </summary>
        private void UpdateFpsMonitorInferenceInfo()
        {
            if (FpsMonitor == null || PaddleOCR == null)
                return;

            TextDetector detector = PaddleOCR.Detector;
            if (detector != null)
            {
                int be = detector.DnnBackend;
                int tgt = detector.DnnTarget;
                FpsMonitor.Add("dnnBackend", MultiBackendDnn.GetBackendDisplayString(be));
                FpsMonitor.Add("dnnTarget", MultiBackendDnn.GetTargetDisplayString(tgt));
            }
            else
            {
                FpsMonitor.Add("dnnBackend", "-");
                FpsMonitor.Add("dnnTarget", "-");
            }

            FpsMonitor.Add("useAsyncInference", PaddleOCR.UseAsyncInference.ToString());
        }
    }
}

#endif
