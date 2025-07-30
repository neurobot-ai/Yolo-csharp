using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using YoloV11.Models;
using YoloV11.Utils;
using System.Runtime.InteropServices;


namespace YoloV11
{
    /// <summary>
    /// 用于执行 YOLOv11 检测和分割的统一类。
    /// 兼容 .NET Framework 4.6.2。
    /// </summary>
    public class YoloV11 : IDisposable
    {
        private enum ModelType { Detection, Segmentation }

        private readonly InferenceSession _session;
        private readonly ModelType _modelType;
        private readonly bool _isDynamicInputShape;
        private readonly Size _inputImageShape;
        private readonly string[] _classNames;
        private readonly Scalar[] _classColors;

        public IReadOnlyList<string> ClassNames { get { return _classNames; } }
        public IReadOnlyList<Scalar> ClassColors { get { return _classColors; } }

        /// <summary>
        /// 构造函数，加载模型并自动确定任务类型。
        /// </summary>
        /// <param name="modelPath">ONNX模型的路径。</param>
        /// <param name="labelsPath">类别标签文件的路径。</param>
        /// <param name="useGpu">是否使用GPU进行推理。</param>
        public YoloV11(string modelPath, string labelsPath, bool useGpu = false)
        {
            var stopwatch = Stopwatch.StartNew();

            SessionOptions options = new SessionOptions();
            options.IntraOpNumThreads = Math.Min(6, Environment.ProcessorCount);
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            if (useGpu)
            {
                try
                {
                    options.AppendExecutionProvider_CUDA();
                    Console.WriteLine("[INFO] Using GPU (CUDA) for YOLOv11 inference.");
                }
                catch (Exception)
                {
                    Console.WriteLine("[WARNING] GPU not available or ONNX Runtime not built with CUDA support. Falling back to CPU.");
                    Console.WriteLine("[INFO] Using CPU for YOLOv11 inference.");
                }
            }
            else
            {
                Console.WriteLine("[INFO] Using CPU for YOLOv11 inference.");
            }

            _session = new InferenceSession(modelPath, options);

            // --- Input Node ---
            var inputMetadata = _session.InputMetadata.Values.First();
            var inputShape = inputMetadata.Dimensions;
            if (inputShape.Length != 4)
            {
                throw new ArgumentException("Model input is not 4D! Expected [N, C, H, W].");
            }
            _isDynamicInputShape = inputShape[2] == -1 || inputShape[3] == -1;
            _inputImageShape = _isDynamicInputShape ? new Size(640, 640) : new Size(inputShape[3], inputShape[2]);

            // --- Output Nodes & Model Type Detection ---
            int numOutputNodes = _session.OutputMetadata.Count;
            _modelType = numOutputNodes == 2 ? ModelType.Segmentation : ModelType.Detection;
            if (numOutputNodes < 1 || numOutputNodes > 2)
            {
                throw new ArgumentException(string.Format("Unsupported model type. Expected 1 (Detection) or 2 (Segmentation) output nodes, but got {0}", numOutputNodes));
            }

            // --- Load Class Data ---
            _classNames = File.ReadAllLines(labelsPath).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
            _classColors = CvUtils.GenerateColors(_classNames);

            stopwatch.Stop();
            Console.WriteLine(string.Format("[INFO] YOLOv11 loaded: {0} in {1} ms", Path.GetFileName(modelPath), stopwatch.ElapsedMilliseconds));
            Console.WriteLine(string.Format("       Model Type:    {0}", _modelType));
            Console.WriteLine(string.Format("       Input shape:   {0} {1}", _inputImageShape, (_isDynamicInputShape ? "(dynamic)" : "")));
            Console.WriteLine(string.Format("       #Classes:      {0}", _classNames.Length));
        }

        public List<Result> Predict(Mat image, float confThreshold = 0.40f, float iouThreshold = 0.45f)
        {
            // --- Preprocess ---
            float ratio;
            Size newUnpad;
            Mat letterboxImg = CvUtils.LetterBox(image, _inputImageShape, out newUnpad, out ratio);
            var inputTensor = Preprocess(letterboxImg);

            // --- Inference ---
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_session.InputMetadata.Keys.First(), inputTensor)
            };

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = _session.Run(inputs))
            {
                // --- Postprocess ---
                return Postprocess(outputs, image.Size(), letterboxImg.Size(), confThreshold, iouThreshold);
            }
        }

        private DenseTensor<float> Preprocess(Mat image)
        {
            Mat rgbImage = new Mat();
            Cv2.CvtColor(image, rgbImage, ColorConversionCodes.BGR2RGB);

            Mat floatImage = new Mat();
            rgbImage.ConvertTo(floatImage, MatType.CV_32FC3, 1.0 / 255.0);

            var tensor = new DenseTensor<float>(new[] { 1, 3, floatImage.Height, floatImage.Width });

            int channelSize = floatImage.Height * floatImage.Width;
            float[] data = new float[channelSize * 3];

            // --- 最终正确方案：使用 Marshal.Copy 进行内存复制 ---
            // 之前所有尝试的 GetArray() 和 Get<T>() 都是错误的，请删除它们。

            // 1. 获取 Mat 对象的内存起始地址 (IntPtr)
            IntPtr sourcePtr = floatImage.Data;

            // 2. 将 Mat 内存中的数据，从头到尾复制到 C# 的 float[] 数组中
            Marshal.Copy(sourcePtr, data, 0, data.Length);
            // --- 修改结束 ---

            // 后续的循环逻辑保持不变，它负责将 HWC 格式的 data 数组重排为 NCHW 格式的 tensor
            for (int y = 0; y < floatImage.Height; y++)
            {
                for (int x = 0; x < floatImage.Width; x++)
                {
                    int baseIdx = (y * floatImage.Width + x) * 3;
                    tensor[0, 0, y, x] = data[baseIdx];     // R
                    tensor[0, 1, y, x] = data[baseIdx + 1]; // G
                    tensor[0, 2, y, x] = data[baseIdx + 2]; // B
                }
            }

            rgbImage.Dispose();
            floatImage.Dispose();
            return tensor;
        }

        // 在 YoloV11.cs 文件中找到并完整替换这个方法
        private List<Result> Postprocess(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs, Size origSize, Size letterboxSize, float confThreshold, float iouThreshold)
        {
            var output0 = outputs.First().AsTensor<float>();

            var shape = output0.Dimensions.ToArray();
            int numFeatures = shape[1];
            int numDetections = shape[2];

            Mat outputMat = new Mat(numFeatures, numDetections, MatType.CV_32F);
            float[] data = output0.ToArray();
            Marshal.Copy(data, 0, outputMat.Data, data.Length);
            Mat transposedOutput = outputMat.T();

            int numClasses;
            int maskCoeffsCount = 0;

            if (_modelType == ModelType.Segmentation)
            {
                maskCoeffsCount = 32;
                numClasses = numFeatures - 4 - maskCoeffsCount;
            }
            else // Detection
            {
                numClasses = numFeatures - 4;
            }

            var boxes = new List<Rect>();
            var confs = new List<float>();
            var classIds = new List<int>();
            var maskCoeffsList = new List<Mat>();

            for (int i = 0; i < transposedOutput.Rows; i++)
            {
                Mat row = transposedOutput.Row(i);
                Mat scoresMat = row.ColRange(4, 4 + numClasses);

                double minConf, maxConf;
                Point minLoc, maxLoc;
                Cv2.MinMaxLoc(scoresMat, out minConf, out maxConf, out minLoc, out maxLoc);

                if (maxConf >= confThreshold)
                {
                    float cx = row.At<float>(0, 0);
                    float cy = row.At<float>(0, 1);
                    float w = row.At<float>(0, 2);
                    float h = row.At<float>(0, 3);
                    boxes.Add(new Rect((int)(cx - w / 2), (int)(cy - h / 2), (int)w, (int)h));
                    confs.Add((float)maxConf);
                    classIds.Add(maxLoc.X);

                    if (_modelType == ModelType.Segmentation)
                    {
                        // 使用 .Clone() 来创建数据的独立副本，确保后续使用的安全性
                        maskCoeffsList.Add(row.ColRange(4 + numClasses, numFeatures).Clone());
                    }
                }
            }

            if (boxes.Count == 0)
            {
                transposedOutput.Dispose();
                outputMat.Dispose();
                return new List<Result>();
            }

            int[] nmsIndices;
            CvDnn.NMSBoxes(boxes, confs, confThreshold, iouThreshold, out nmsIndices);

            var finalResults = new List<Result>();
            if (nmsIndices.Length == 0)
            {
                transposedOutput.Dispose();
                outputMat.Dispose();
                return finalResults;
            }

            Mat prototypeMasksMat = null;
            int maskH = 0;
            int maskW = 0;

            if (_modelType == ModelType.Segmentation)
            {
                var output1 = outputs.Last().AsTensor<float>();
                maskH = output1.Dimensions[2];
                maskW = output1.Dimensions[3];

                prototypeMasksMat = new Mat(maskCoeffsCount, maskH * maskW, MatType.CV_32F);
                float[] protoData = output1.ToArray();
                Marshal.Copy(protoData, 0, prototypeMasksMat.Data, protoData.Length);
            }

            foreach (int idx in nmsIndices)
            {
                BoundingBox scaledBox = CvUtils.ScaleCoords(letterboxSize, boxes[idx], origSize);

                var res = new Result
                {
                    Box = scaledBox,
                    Confidence = confs[idx],
                    ClassId = classIds[idx]
                };

                if (_modelType == ModelType.Segmentation)
                {
                    // 使用 using 语句确保临时 Mat 对象被正确释放
                    using (Mat maskCoeffsMat = maskCoeffsList[idx])
                    using (Mat combinedMask = new Mat())
                    {
                        // 1. 矩阵乘法: [1, 32] * [32, maskH * maskW] -> [1, maskH * maskW]
                        Cv2.Gemm(maskCoeffsMat, prototypeMasksMat, 1, new Mat(), 0, combinedMask, GemmFlags.None);

                        // 2. [关键修正] 将 [1, maskH * maskW] 的长条数据重塑为 [maskH, maskW] 的图像掩码
                        Mat reshapedMask = combinedMask.Reshape(1, maskH);

                        // 3. 应用 Sigmoid 函数将值缩放到 0-1 之间
                        CvUtils.Sigmoid(reshapedMask, reshapedMask);

                        // 4. 将掩码缩放、裁剪并二值化
                        res.Mask = ProcessMask(reshapedMask, letterboxSize, origSize, new Rect(scaledBox.X, scaledBox.Y, scaledBox.Width, scaledBox.Height));

                        // 5. 释放重塑后的掩码
                        reshapedMask.Dispose();
                    }
                }

                finalResults.Add(res);
            }

            if (prototypeMasksMat != null) prototypeMasksMat.Dispose();
            transposedOutput.Dispose();
            outputMat.Dispose();

            return finalResults;
        }
        private Mat ProcessMask(Mat maskProto, Size letterboxSize, Size originalSize, Rect roi)
        {
            float gain = Math.Min((float)letterboxSize.Height / originalSize.Height, (float)letterboxSize.Width / originalSize.Width);
            int padW = (int)Math.Round((letterboxSize.Width - originalSize.Width * gain) / 2.0f);
            int padH = (int)Math.Round((letterboxSize.Height - originalSize.Height * gain) / 2.0f);

            int maskW = maskProto.Cols;
            int maskH = maskProto.Rows;

            int mask_x1 = (int)((float)padW / letterboxSize.Width * maskW);
            int mask_y1 = (int)((float)padH / letterboxSize.Height * maskH);
            int mask_x2 = maskW - mask_x1;
            int mask_y2 = maskH - mask_y1;

            Rect cropRect = new Rect(mask_x1, mask_y1, mask_x2 - mask_x1, mask_y2 - mask_y1);
            using (Mat croppedMask = new Mat(maskProto, cropRect))
            using (Mat resizedMask = new Mat())
            {
                Cv2.Resize(croppedMask, resizedMask, originalSize, 0, 0, InterpolationFlags.Linear);

                Mat finalBinaryMask = Mat.Zeros(originalSize, MatType.CV_8U);
                Rect validRoi = roi.Intersect(new Rect(0, 0, originalSize.Width, originalSize.Height));

                if (validRoi.Width > 0 && validRoi.Height > 0)
                {
                    using (Mat binaryMask = new Mat())
                    {
                        Cv2.Threshold(resizedMask[validRoi], binaryMask, 0.5f, 255, ThresholdTypes.Binary);
                        binaryMask.ConvertTo(binaryMask, MatType.CV_8U);
                        binaryMask.CopyTo(finalBinaryMask[validRoi]);
                    }
                }
                return finalBinaryMask;
            }
        }

        public void DrawResults(Mat image, IReadOnlyList<Result> results, float maskAlpha = 0.4f)
        {
            foreach (var res in results)
            {
                if (res.Confidence < 0.25) continue;

                var color = _classColors[res.ClassId % _classColors.Length];

                if (res.Mask != null && !res.Mask.Empty())
                {
                    using (Mat coloredMask = new Mat())
                    {
                        Cv2.CvtColor(res.Mask, coloredMask, ColorConversionCodes.GRAY2BGR);
                        coloredMask.SetTo(color, res.Mask);
                        Cv2.AddWeighted(image, 1.0, coloredMask, maskAlpha, 0, image);
                    }
                }

                Cv2.Rectangle(image, new Rect(res.Box.X, res.Box.Y, res.Box.Width, res.Box.Height), color, 2);

                string label = string.Format("{0} {1:P0}", _classNames[res.ClassId], res.Confidence);
                var fontFace = HersheyFonts.HersheySimplex;
                int baseline;
                Size labelSize = Cv2.GetTextSize(label, fontFace, 0.5, 1, out baseline);

                int top = Math.Max(res.Box.Y, labelSize.Height + 5);
                Cv2.Rectangle(image, new Point(res.Box.X, top - labelSize.Height - 5), new Point(res.Box.X + labelSize.Width, top), color, -1);
                Cv2.PutText(image, label, new Point(res.Box.X + 2, top - 3), fontFace, 0.5, Scalar.White, 1, LineTypes.AntiAlias);
            }
        }

        public void Dispose()
        {
            if (_session != null)
            {
                _session.Dispose();
            }
        }
    }
}