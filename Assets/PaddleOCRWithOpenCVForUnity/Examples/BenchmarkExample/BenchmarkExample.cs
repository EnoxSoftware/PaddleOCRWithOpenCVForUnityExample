#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System;
using System.Linq;
using System.Threading;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.DnnModule;
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
    /// Runs <see cref="PaddleOCRComponent"/> OCR on an Inspector-assigned <see cref="Texture2D"/> and
    /// shows detection boxes and recognition text overlaid on <see cref="RawImage"/>.
    /// </summary>
    /// <remarks>
    /// Add <see cref="PaddleOCRComponent"/> on the same GameObject and configure model paths.
    /// Receive OCR completion via <see cref="PaddleOCRComponent.WorkCompleted"/> (UnityEvent) and
    /// <see cref="OnOCRWorkCompleted"/> (manual Inspector wiring also supported).
    /// Recognition text is shown in <see cref="RecognitionResultField"/>.
    /// </remarks>
    [RequireComponent(typeof(PaddleOCRComponent))]
    public class BenchmarkExample : MonoBehaviour
    {
        // Public Fields
        [Header("PaddleOCR")]
        [Tooltip("OCR inference component. Assign the PaddleOCRComponent on the same GameObject in the Inspector.")]
        public PaddleOCRComponent PaddleOCR;

        [Header("Input")]
        [Tooltip("Texture to OCR. Assign in the Inspector.")]
        public Texture2D InputTexture;

        [Header("Output")]
        [Tooltip("Display target for OCR results.")]
        public RawImage ResultPreview;

        [Tooltip("TMP InputField for recognition results.")]
        public TMP_InputField RecognitionResultField;

        [Tooltip("Run OCR automatically on Start.")]
        public bool RunOCROnStart = true;

        [Tooltip("Print OCR results to the Unity console.")]
        public bool PrintResultToConsole = true;

        [Header("FpsMonitor")]
        [Tooltip("FPS / inference info display. Uses the component on the same GameObject when unset.")]
        public FpsMonitor FpsMonitor;

        // Private Fields
        private Mat _rgbaMat;
        private Mat _bgrMat;
        private Mat _displayBgrMat;
        private Texture2D _outputTexture;
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private readonly System.Diagnostics.Stopwatch _inferenceStopwatch = new System.Diagnostics.Stopwatch();
        private bool _inferenceTimingActive;

        // Unity Lifecycle Methods

        /// <summary>
        /// On startup, waits for <see cref="PaddleOCRComponent"/> initialization, then starts OCR when <see cref="RunOCROnStart"/> is enabled.
        /// </summary>
        private async void Start()
        {
            OpenCVDebug.SetDebugMode(true);

            if (PaddleOCR == null)
            {
                Debug.LogError($"{nameof(BenchmarkExample)}: {nameof(PaddleOCRComponent)} is not assigned.");
                return;
            }

            if (FpsMonitor == null)
            {
                Debug.LogError($"{nameof(BenchmarkExample)}: {nameof(FpsMonitor)} is not assigned.");
                return;
            }

            // AutoInitializeOnStart … InitializeInferenceAsync starts automatically in PaddleOCRComponent.Awake (Inspector default).
            await PaddleOCR.WaitForInitializationAsync();
            if (!PaddleOCR.IsInitialized)
            {
                Debug.LogWarning($"{nameof(BenchmarkExample)}: {nameof(PaddleOCRComponent)} is not initialized.");
                return;
            }

            UpdateFpsMonitorInferenceMs("-");

            if (RunOCROnStart)
                RunOCRAsync();
        }

        /// <summary>
        /// Removes event listeners, disposes Mats/textures, and releases the cancellation token.
        /// </summary>
        private void OnDestroy()
        {
            if (PaddleOCR != null)
                PaddleOCR.WorkCompleted.RemoveListener(OnOCRWorkCompleted);

            _cts.Cancel();
            _cts.Dispose();

            _rgbaMat?.Dispose(); _rgbaMat = null;
            _bgrMat?.Dispose(); _bgrMat = null;
            _displayBgrMat?.Dispose(); _displayBgrMat = null;

            if (_outputTexture != null)
                Texture2D.Destroy(_outputTexture);
            _outputTexture = null;

            OpenCVDebug.SetDebugMode(false);
        }

        // Public Methods
        /// <summary>
        /// Called from <see cref="PaddleOCRComponent.WorkCompleted"/> (Inspector: Work Completed → this method).
        /// On success, updates the preview and <see cref="RecognitionResultField"/>.
        /// </summary>
        /// <param name="kind">Completion kind.</param>
        /// <param name="errorMessage">Message on failure/cancel (often empty on success).</param>
        public void OnOCRWorkCompleted(WorkCompletionKind kind, string errorMessage)
        {
            if (PaddleOCR == null || !PaddleOCR.IsInitialized)
                return;

            CompleteInferenceTiming();

            if (kind != WorkCompletionKind.Succeeded)
            {
                if (kind == WorkCompletionKind.Faulted && !string.IsNullOrEmpty(errorMessage))
                    Debug.LogWarning($"{nameof(BenchmarkExample)} OCR: {errorMessage}");
                return;
            }

            UpdateRecognitionResultField();
            RefreshPreviewWithOCROverlay();
            UpdateFpsMonitorInferenceInfo();
        }

        /// <summary>
        /// Called after inference UI settings change (Inspector: InferenceSettingsChanged → this method).
        /// </summary>
        public void OnPaddleOCRInferenceSettingsChanged()
        {
            UpdateFpsMonitorInferenceInfo();
            RerunOCRAfterInferenceSettingsChanged();
        }

        /// <summary>
        /// Returns to the main menu. Wire from the Back button in the Inspector.
        /// </summary>
        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("PaddleOCRWithOpenCVForUnityExample");
        }

        /// <summary>
        /// Re-runs OCR when the Run OCR button is pressed. Wire in the Inspector.
        /// </summary>
        public void OnRunOCRButtonClick()
        {
            RunOCRAsync();
        }

        // Private Methods

        /// <summary>
        /// Converts the input texture to a BGR <see cref="Mat"/> and submits OCR via
        /// <see cref="PaddleOCRComponent.Submit(OpenCVForUnity.CoreModule.Mat)"/>.
        /// Requires <see cref="PaddleOCRComponent.IsInitialized"/> (see <see cref="Start"/>).
        /// Completion is received in <see cref="OnOCRWorkCompleted"/> (<see cref="PaddleOCRComponent.WorkCompleted"/>).
        /// </summary>
        private void RunOCRAsync()
        {
            if (InputTexture == null)
            {
                Debug.LogError($"{nameof(BenchmarkExample)}: Input Texture2D is not assigned.");
                return;
            }

            if (_cts.IsCancellationRequested || PaddleOCR == null || !PaddleOCR.IsInitialized)
                return;

            UpdateFpsMonitorInferenceInfo();

            if (!LoadImageMats())
                return;

            ShowSourcePreview();

            BeginInferenceTiming();

            if (!PaddleOCR.Submit(_bgrMat))
            {
                Debug.LogWarning($"{nameof(BenchmarkExample)}: Submit failed (not initialized?).");
                _inferenceTimingActive = false;
                UpdateFpsMonitorInferenceMs("-");
                return;
            }
        }

        /// <summary>
        /// Re-runs OCR after inference settings change when an input texture is assigned.
        /// </summary>
        private void RerunOCRAfterInferenceSettingsChanged()
        {
            if (InputTexture == null || PaddleOCR == null || _cts.IsCancellationRequested)
                return;

            RunOCRAsync();
        }

        /// <summary>
        /// Loads <see cref="InputTexture"/> into an RGBA <see cref="Mat"/> and prepares BGR Mat and a display clone.
        /// </summary>
        /// <returns>true when the input texture is valid.</returns>
        private bool LoadImageMats()
        {
            if (InputTexture == null)
                return false;

            int width = InputTexture.width;
            int height = InputTexture.height;

            if (_rgbaMat == null
                || _rgbaMat.cols() != width
                || _rgbaMat.rows() != height)
            {
                _rgbaMat?.Dispose();
                _rgbaMat = new Mat(height, width, CvType.CV_8UC4);
            }

            if (_bgrMat == null
                || _bgrMat.cols() != width
                || _bgrMat.rows() != height)
            {
                _bgrMat?.Dispose();
                _bgrMat = new Mat(height, width, CvType.CV_8UC3);
            }

            OpenCVMatUtils.Texture2DToMat(InputTexture, _rgbaMat);
            Imgproc.cvtColor(_rgbaMat, _bgrMat, Imgproc.COLOR_RGBA2BGR);

            _displayBgrMat?.Dispose();
            _displayBgrMat = _bgrMat.clone();
            return true;
        }

        /// <summary>
        /// Shows the source image (no detection overlay) on <see cref="ResultPreview"/> before OCR runs.
        /// </summary>
        private void ShowSourcePreview()
        {
            if (_displayBgrMat == null)
                return;

            UpdateResultPreview(_displayBgrMat);
        }

        /// <summary>
        /// Fetches the latest OCR result, overlays detection boxes and text, and updates <see cref="ResultPreview"/>.
        /// </summary>
        private void RefreshPreviewWithOCROverlay()
        {
            if (_displayBgrMat == null || _bgrMat == null)
                return;

            _bgrMat.copyTo(_displayBgrMat);

            if (!PaddleOCR.TryGetLatestResultViews(out Mat[] detView, out Mat[] clsView, out Mat[] recView))
                return;

            PaddleOCRPipelineUtility.VisualizeOCRResults(
                _displayBgrMat,
                detView,
                clsView,
                recView,
                printResult: PrintResultToConsole,
                isRGB: false);

            UpdateResultPreview(_displayBgrMat);
        }

        /// <summary>
        /// Shows the latest OCR recognition text in <see cref="RecognitionResultField"/>.
        /// </summary>
        private void UpdateRecognitionResultField()
        {
            if (RecognitionResultField == null || PaddleOCR == null)
                return;

            if (!PaddleOCR.TryGetLatestParsedResult(out PaddleOCRParsedResult parsed))
                return;

            var recognitions = parsed.Recognitions;
            string text = recognitions == null || recognitions.Count == 0
                ? string.Empty
                : string.Join("\n", recognitions.Select(r => r.text));

            RecognitionResultField.text = text;
            RecognitionResultField.stringPosition = 0;
            RecognitionResultField.caretPosition = 0;
            if (RecognitionResultField.verticalScrollbar != null)
                RecognitionResultField.verticalScrollbar.value = 0;
        }

        /// <summary>
        /// Converts a BGR <see cref="Mat"/> to RGB <see cref="Texture2D"/> and displays it on <see cref="ResultPreview"/>.
        /// </summary>
        /// <param name="bgrMat">BGR image to display.</param>
        private void UpdateResultPreview(Mat bgrMat)
        {
            if (ResultPreview == null || bgrMat == null)
                return;

            using (Mat rgbMat = new Mat())
            {
                Imgproc.cvtColor(bgrMat, rgbMat, Imgproc.COLOR_BGR2RGB);

                if (_outputTexture == null
                    || _outputTexture.width != rgbMat.cols()
                    || _outputTexture.height != rgbMat.rows())
                {
                    if (_outputTexture != null)
                        Texture2D.Destroy(_outputTexture);

                    _outputTexture = new Texture2D(rgbMat.cols(), rgbMat.rows(), TextureFormat.RGB24, false);
                }

                OpenCVMatUtils.MatToTexture2D(rgbMat, _outputTexture);
            }

            ResultPreview.texture = _outputTexture;

            AspectRatioFitter fitter = ResultPreview.GetComponent<AspectRatioFitter>();
            if (fitter != null)
                fitter.aspectRatio = (float)_outputTexture.width / _outputTexture.height;
        }

        /// <summary>
        /// Starts OCR inference timing and sets <see cref="FpsMonitor"/> inferenceMs to "...".
        /// </summary>
        private void BeginInferenceTiming()
        {
            _inferenceTimingActive = true;
            _inferenceStopwatch.Restart();
            UpdateFpsMonitorInferenceMs("...");
        }

        /// <summary>
        /// Stops OCR inference timing and shows elapsed milliseconds on <see cref="FpsMonitor"/>.
        /// </summary>
        private void CompleteInferenceTiming()
        {
            if (!_inferenceTimingActive)
                return;

            _inferenceTimingActive = false;
            _inferenceStopwatch.Stop();
            UpdateFpsMonitorInferenceMs($"{_inferenceStopwatch.ElapsedMilliseconds} ms");
        }

        /// <summary>
        /// Shows DNN backend, target, and async inference settings on <see cref="FpsMonitor"/>.
        /// </summary>
        private void UpdateFpsMonitorInferenceInfo()
        {
            if (PaddleOCR == null)
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

        /// <summary>
        /// Shows inference time (ms) or a placeholder string on <see cref="FpsMonitor"/>.
        /// </summary>
        /// <param name="value">Display string (e.g. "123 ms", "-", "...").</param>
        private void UpdateFpsMonitorInferenceMs(string value)
        {
            FpsMonitor.Add("inferenceMs", value);
        }
    }
}

#endif
