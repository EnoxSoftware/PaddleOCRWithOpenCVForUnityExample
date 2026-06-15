#if !UNITY_WSA_10_0 && NET_STANDARD_2_1 && !OPENCV_DONT_USE_UNSAFE_CODE

using System.Linq;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.UnityIntegration;
using OpenCVForUnity.UnityIntegration.Runner;
using PaddleOCRWithOpenCVForUnity;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace PaddleOCRWithOpenCVForUnityExample
{
    /// <summary>
    /// Minimal PaddleOCR tutorial example.
    /// Runs OCR via <see cref="PaddleOCRComponent.Submit(Texture2D)"/> and
    /// receives results through an Inspector-wired <see cref="PaddleOCRParsedResultUnityEvent"/>.
    /// </summary>
    /// <remarks>
    /// Basic API flow:
    /// <list type="number">
    /// <item><description>(Optional) Load models with <see cref="PaddleOCRComponent.InitializeInferenceAsync"/> and set <see cref="PaddleOCRComponent.IsInitialized"/> to true.</description></item>
    /// <item><description>Submit images to the Runner with <see cref="PaddleOCRComponent.Submit(Texture2D)"/> or <see cref="PaddleOCRComponent.Submit(OpenCVForUnity.CoreModule.Mat)"/>.</description></item>
    /// <item><description>After inference completes, <see cref="OnOCRResult"/> is called from Inspector-wired <see cref="PaddleOCRComponent.WorkCompletedWithParsedResult"/> (<see cref="PaddleOCRParsedResultUnityEvent"/>).</description></item>
    /// <item><description>Read recognition strings from <see cref="PaddleOCRParsedResult.Recognitions"/>.</description></item>
    /// </list>
    /// Setup:
    /// 1. Add <see cref="PaddleOCRComponent"/> on the same GameObject and configure model paths in the Inspector.
    /// 2. Assign this script's <see cref="OnOCRResult"/> to PaddleOCR Work Completed With Parsed Result (pre-wired in the scene).
    /// 3. Assign the source <see cref="Texture2D"/> on this script's Input field.
    /// 4. Assign a TMP InputField to Recognition Result Field for output.
    /// 5. Press Play: one OCR run in Start; on completion, preview with boxes and TMP field are updated (console via Print Result To Console).
    /// <para>
    /// <see cref="PaddleOCRComponent.DisposeInferenceAsync"/> runs automatically when <see cref="PaddleOCRComponent"/> is destroyed; this example does not call it.
    /// <see cref="OnDestroy"/> only destroys runtime-created preview <see cref="Texture2D"/> instances.
    /// </para>
    /// </remarks>
    [RequireComponent(typeof(PaddleOCRComponent))]
    public class TutorialExample : MonoBehaviour
    {
        // Public Fields
        // PaddleOCRComponent … OCR entry point; model paths and inference settings are on its Inspector.

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

        [Tooltip("Print OCR results to the Unity console.")]
        public bool PrintResultToConsole = true;

        // Unity Lifecycle Methods

        private async void Start()
        {
            ShowInputPreview();

            if (PaddleOCR == null)
            {
                Debug.LogError($"{nameof(TutorialExample)}: {nameof(PaddleOCRComponent)} is not assigned.");
                return;
            }

            if (InputTexture == null)
            {
                Debug.LogWarning($"{nameof(TutorialExample)}: Input Texture2D is not assigned.");
                return;
            }

            // --- 1. Initialization ---
            // Submit only when IsInitialized is true.
            // AutoInitializeOnStart … InitializeInferenceAsync starts in PaddleOCRComponent.Awake (default for this example).
            // When false, await InitializeInferenceAsync yourself.
            if (PaddleOCR.AutoInitializeOnStart && PaddleOCR.IsInitializing)
                await PaddleOCR.WaitForInitializationAsync();
            else
            {
                await PaddleOCR.InitializeInferenceAsync();
            }

            if (!PaddleOCR.IsInitialized)
            {
                Debug.LogWarning($"{nameof(TutorialExample)}: {nameof(PaddleOCRComponent)} is not initialized. OCR skipped.");
                return;
            }

            // --- 2. Image submission (starts async inference) ---
            // Submit returns true/false immediately (whether submission was attempted). No recognition text yet.
            // After completion, results can also be received via:
            //   WorkCompleted … completion kind and error message only (Inspector)
            //   WorkCompletedCode … subscribe to WorkCompletion<Mat> from code
            //   TryGetLatestParsedResult / TryGetLatestResult / TryGetLatestResultViews … poll for latest results
            // This example uses WorkCompletedWithParsedResult → OnOCRResult (success only by default;
            // set InvokeParsedResultOnSuccess = false on PaddleOCRComponent to also receive failure/cancel).
            // For Mat input use BGR 8UC3. Texture2D is converted to BGR internally; temporary Mat is disposed inside Submit.
            PaddleOCR.Submit(InputTexture);
        }

        /// <summary>Releases runtime-created preview <see cref="Texture2D"/> instances.</summary>
        private void OnDestroy()
        {
            if (ResultPreview == null)
                return;

            Texture2D texture = ResultPreview.texture as Texture2D;
            if (texture != null && texture != InputTexture)
                Texture2D.Destroy(texture);
        }

        // Public Methods

        /// <summary>
        /// Called from <see cref="PaddleOCRComponent.WorkCompletedWithParsedResult"/> (wire in Inspector).
        /// On success, overlays <see cref="PaddleOCRParsedResult"/> on <see cref="ResultPreview"/> and
        /// shows recognition text in <see cref="RecognitionResultField"/> (console via <see cref="PrintResultToConsole"/>).
        /// </summary>
        public void OnOCRResult(PaddleOCRParsedResult result)
        {
            // --- 3. Receiving results ---
            // PaddleOCRComponent Inspector:
            //   Work Completed With Parsed Result → this method (PaddleOCRParsedResultUnityEvent)
            // Kind … Succeeded / Failed / Cancelled, etc. Recognitions is valid only on success.
            // Failure/cancel branches run only when PaddleOCRComponent.InvokeParsedResultOnSuccess is false.

            if (result.Kind != WorkCompletionKind.Succeeded)
            {
                if (!string.IsNullOrEmpty(result.ErrorMessage))
                    Debug.LogWarning($"{nameof(TutorialExample)} OCR {result.Kind}: {result.ErrorMessage}");
                else
                    Debug.Log($"{nameof(TutorialExample)} OCR {result.Kind}");
                return;
            }

            // Recognitions … list of (text, score). Detections → result.Detections, classifications → result.Classifications.
            var recognitions = result.Recognitions;
            string text = recognitions == null || recognitions.Count == 0
                ? string.Empty
                : string.Join("\n", recognitions.Select(r => r.text));

            if (RecognitionResultField != null)
            {
                RecognitionResultField.text = text;
                RecognitionResultField.stringPosition = 0;
                RecognitionResultField.caretPosition = 0;
                if (RecognitionResultField.verticalScrollbar != null)
                    RecognitionResultField.verticalScrollbar.value = 0;
            }

            UpdateResultPreviewWithOCROverlay(result);
        }

        /// <summary>
        /// Returns to the main menu. Wire from the Back button in the Inspector.
        /// </summary>
        public void OnBackButtonClick()
        {
            SceneManager.LoadScene("PaddleOCRWithOpenCVForUnityExample");
        }

        // Private Methods
        /// <summary>
        /// Shows Inspector-assigned <see cref="InputTexture"/> on <see cref="ResultPreview"/>.
        /// </summary>
        private void ShowInputPreview()
        {
            if (ResultPreview == null || InputTexture == null)
                return;

            ResultPreview.texture = InputTexture;

            AspectRatioFitter fitter = ResultPreview.GetComponent<AspectRatioFitter>();
            if (fitter != null)
                fitter.aspectRatio = (float)InputTexture.width / InputTexture.height;
        }

        /// <summary>
        /// Converts <see cref="InputTexture"/> to an RGB <see cref="Mat"/>, overlays OCR results,
        /// writes back to <see cref="Texture2D"/>, and displays on <see cref="ResultPreview"/>.
        /// </summary>
        private void UpdateResultPreviewWithOCROverlay(PaddleOCRParsedResult result)
        {
            if (InputTexture == null || ResultPreview == null)
                return;

            int width = InputTexture.width;
            int height = InputTexture.height;

            Texture2D previousTexture = ResultPreview.texture as Texture2D;
            if (previousTexture != null && previousTexture != InputTexture)
                Texture2D.Destroy(previousTexture);

            using (Mat rgbMat = new Mat(height, width, CvType.CV_8UC3))
            {
                OpenCVMatUtils.Texture2DToMat(InputTexture, rgbMat);

                // --- 4. (Optional) Visualize results ---
                // PaddleOCRPipelineUtility.VisualizeOCRResults … draws boxes and text on the Mat.
                // isRGB: true treats rgbMat as RGB order (from Texture2D).
                PaddleOCRPipelineUtility.VisualizeOCRResults(
                    rgbMat,
                    result,
                    printResult: PrintResultToConsole,
                    isRGB: true);

                Texture2D previewTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
                OpenCVMatUtils.MatToTexture2D(rgbMat, previewTexture);

                ResultPreview.texture = previewTexture;

                AspectRatioFitter fitter = ResultPreview.GetComponent<AspectRatioFitter>();
                if (fitter != null)
                    fitter.aspectRatio = (float)width / height;
            }
        }
    }
}

#endif
