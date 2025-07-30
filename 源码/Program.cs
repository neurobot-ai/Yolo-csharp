using OpenCvSharp;
using System;
using System.IO; // Required for file and directory operations
using System.Linq; // Required for filtering file extensions
using YoloV11; // Assuming you used the recommended namespace

class Program
{
    static void Main(string[] args)
    {
        // --- Configuration ---
        // Set your model, source, and destination paths here.
        const string modelPath = @"D:\\Detection-Onnx\\Model\\ball.onnx";
        const string labelsPath = @"D:\\Detection-Onnx\\Dataset\\names.txt";
        
        // Directory containing the images you want to process.
        const string datasetDir = @"D:\\Detection-Onnx\\Dataset\\myball"; 
        
        // Directory where the processed images will be saved.
        const string resultDir = "Run"; 
        
        const bool useGpu = true;

        // Thresholds
        const float confThreshold = 0.4f;
        const float iouThreshold = 0.45f;

        // --- Main Logic ---
        try
        {
            // 1. Initialize the YOLOv11 detector once.
            using (var yolo = new YoloV11.YoloV11(modelPath, labelsPath, useGpu))
            {
                // 2. Ensure the output directory exists. If not, create it.
                if (!Directory.Exists(resultDir))
                {
                    Directory.CreateDirectory(resultDir);
                }

                // 3. Define the allowed image file extensions.
                var allowedExtensions = new string[] { ".jpg", ".jpeg", ".png", ".bmp" };

                // 4. Get all image files from the source directory.
                var imageFiles = Directory.EnumerateFiles(datasetDir)
                                          .Where(file => allowedExtensions.Contains(Path.GetExtension(file).ToLower()))
                                          .ToList();

                if (imageFiles.Count == 0)
                {
                    Console.WriteLine($"No images found in the specified directory: {datasetDir}");
                    return;
                }
                
                Console.WriteLine($"Found {imageFiles.Count} images to process...");

                // 5. Loop through each image file.
                foreach (var imagePath in imageFiles)
                {
                    // Load the image
                    using (Mat image = new Mat(imagePath))
                    {
                        if (image.Empty())
                        {
                            Console.WriteLine($"Warning: Could not read image file: {imagePath}");
                            continue; // Skip to the next file
                        }

                        // Perform prediction
                        var results = yolo.Predict(image, confThreshold, iouThreshold);

                        // Draw results on the image
                        yolo.DrawResults(image, results);

                        // Construct the full output path and save the annotated image
                        string fileName = Path.GetFileName(imagePath);
                        string outputPath = Path.Combine(resultDir, fileName);
                        Cv2.ImWrite(outputPath, image);

                        // Logging
                        Console.WriteLine($"Processed: {fileName}, saved to {outputPath}");
                    }
                }
            }
            
            Console.WriteLine("\nProcessing complete.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}