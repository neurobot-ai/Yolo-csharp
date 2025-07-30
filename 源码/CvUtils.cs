using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Linq;
// 引入 Marshal 类所在的命名空间
using System.Runtime.InteropServices;
using YoloV11.Models;

namespace YoloV11.Utils
{
    public static class CvUtils
    {
        public static T Clamp<T>(T val, T low, T high) where T : IComparable<T>
        {
            if (val.CompareTo(low) < 0) return low;
            if (val.CompareTo(high) > 0) return high;
            return val;
        }

        public static Mat LetterBox(Mat image, Size newShape, out Size newUnpad, out float ratio)
        {
            ratio = Math.Min((float)newShape.Height / image.Height, (float)newShape.Width / image.Width);
            int newW = (int)Math.Round(image.Width * ratio);
            int newH = (int)Math.Round(image.Height * ratio);
            newUnpad = new Size(newW, newH);
            using (Mat resized = new Mat())
            {
                Cv2.Resize(image, resized, newUnpad, 0, 0, InterpolationFlags.Linear);
                int dw = newShape.Width - newW;
                int dh = newShape.Height - newH;
                int top = dh / 2;
                int bottom = dh - top;
                int left = dw / 2;
                int right = dw - left;
                Mat outImage = new Mat();
                Cv2.CopyMakeBorder(resized, outImage, top, bottom, left, right, BorderTypes.Constant, new Scalar(114, 114, 114));
                return outImage;
            }
        }

        public static BoundingBox ScaleCoords(Size letterboxShape, Rect coords, Size originalShape, bool clip = true)
        {
            float gain = Math.Min((float)letterboxShape.Height / originalShape.Height, (float)letterboxShape.Width / originalShape.Width);
            int padW = (int)Math.Round((letterboxShape.Width - originalShape.Width * gain) / 2.0f);
            int padH = (int)Math.Round((letterboxShape.Height - originalShape.Height * gain) / 2.0f);
            var ret = new BoundingBox
            {
                X = (int)Math.Round((coords.X - padW) / gain),
                Y = (int)Math.Round((coords.Y - padH) / gain),
                Width = (int)Math.Round(coords.Width / gain),
                Height = (int)Math.Round(coords.Height / gain)
            };
            if (clip)
            {
                ret.X = Clamp(ret.X, 0, originalShape.Width);
                ret.Y = Clamp(ret.Y, 0, originalShape.Height);
                ret.Width = Clamp(ret.Width, 0, originalShape.Width - ret.X);
                ret.Height = Clamp(ret.Height, 0, originalShape.Height - ret.Y);
            }
            return ret;
        }

        public static void Sigmoid(Mat src, Mat dst)
        {
            using (Mat expSrc = new Mat())
            {
                Cv2.Exp(-src, expSrc);
                using (Mat onePlusExpSrc = new Mat())
                {
                    Cv2.Add(expSrc, new Scalar(1.0), onePlusExpSrc);
                    Cv2.Divide(1.0, onePlusExpSrc, dst);
                }
            }
        }

        public static Scalar[] GenerateColors(IReadOnlyList<string> classNames, int seed = 42)
        {
            var rng = new Random(seed);
            var colors = new Scalar[classNames.Count];
            for (int i = 0; i < classNames.Count; i++)
            {
                colors[i] = new Scalar(rng.Next(0, 256), rng.Next(0, 256), rng.Next(0, 256));
            }
            return colors;
        }

        // --- 第一个 ToCvMat 方法（使用 Marshal.Copy 修正） ---
        public static Mat ToCvMat(this Tensor<float> tensor)
        {
            if (tensor.Rank != 4)
                throw new ArgumentException("Tensor must have a rank of 4.");

            var shape = tensor.Dimensions.ToArray();
            int rows = shape[2];
            int cols = shape[3];
            int channels = shape[1];

            Mat mat = new Mat(rows, cols, MatType.CV_32FC(channels));

            float[] matData = new float[rows * cols * channels];
            for (int c = 0; c < channels; c++)
            {
                for (int y = 0; y < rows; y++)
                {
                    for (int x = 0; x < cols; x++)
                    {
                        matData[(y * cols + x) * channels + c] = tensor[0, c, y, x];
                    }
                }
            }

            // 使用 Marshal.Copy 将数据从 C# 数组直接复制到 Mat 的内存指针
            // mat.Data 提供了指向 Mat 内存的 IntPtr
            Marshal.Copy(matData, 0, mat.Data, matData.Length);

            return mat;
        }

        // --- 第二个 ToCvMat 方法（使用 Marshal.Copy 修正） ---
        public static Mat ToCvMat(this Tensor<float> tensor, int newRows, int newCols)
        {
            var mat = new Mat(newRows, newCols, MatType.CV_32F);
            float[] data = tensor.ToArray();

            // 同样使用 Marshal.Copy 进行内存复制
            Marshal.Copy(data, 0, mat.Data, data.Length);

            return mat;
        }
    }
}