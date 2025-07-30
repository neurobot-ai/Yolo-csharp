using OpenCvSharp;

namespace YoloV11.Models
{
    /// <summary>
    /// 表示一个边界框。
    /// </summary>
    public struct BoundingBox
    {
        public int X { get; set; }
        public int Y { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    /// <summary>
    /// 统一的结果结构体，可同时用于检测和分割。
    /// </summary>
    public class Result
    {
        public BoundingBox Box { get; set; }
        public float Confidence { get; set; }
        public int ClassId { get; set; }

        /// <summary>
        /// 对于检测任务，此项为空。
        /// 对于分割任务，这是一个与原始图像大小相同的单通道 (CV_8U) 掩码，
        /// 其中非零像素表示属于该对象的区域。
        /// </summary>
        public Mat Mask { get; set; } = new Mat();

        public Result() { }

        public Result(BoundingBox box, float confidence, int classId, Mat mask)
        {
            Box = box;
            Confidence = confidence;
            ClassId = classId;
            Mask = mask;
        }
    }
}