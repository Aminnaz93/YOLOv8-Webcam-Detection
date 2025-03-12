#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    // 🟢 Ladda in YOLOv8-modellen från ONNX
    std::string modelPath = "/Users/aminnazari/Desktop/Python/Kamera/yolov8n.onnx";  // Ändra till din ONNX-fil
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    // 🔥 Använd GPU om tillgänglig (annars CPU)
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    // 🖥️ Starta webbkameran
    cv::VideoCapture cap(0);  // 0 = första kameran
    if (!cap.isOpened()) {
        std::cerr << "❌ Kunde inte öppna kameran!" << std::endl;
        return -1;
    }

    // 🚀 YOLOv8 inferens-loop
    while (true) {
        cv::Mat frame;
        cap >> frame;  // Läs en bildruta från kameran
        if (frame.empty()) break;

        // 🟢 Förbered input till YOLO
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);

        // 🚀 Kör inferens
        cv::Mat detections = net.forward();

        // 🔍 Tolka resultatet och rita rektanglar
        float* data = (float*)detections.data;
        for (int i = 0; i < detections.total(); i += 7) {
            float confidence = data[i + 2];  // Konfidenstal
            if (confidence > 0.5) {  // Endast detektioner över 50% säkerhet
                int x1 = (int)(data[i + 3] * frame.cols);
                int y1 = (int)(data[i + 4] * frame.rows);
                int x2 = (int)(data[i + 5] * frame.cols);
                int y2 = (int)(data[i + 6] * frame.rows);

                // 🟢 Rita en grön rektangel runt detekterade objektet
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                cv::putText(frame, "Bestick", cv::Point(x1, y1 - 10),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }

        // 🖥️ Visa kameraflödet med YOLO
        cv::imshow("YOLOv8 C++ Webcam Detection", frame);

        // 🚪 Avsluta med 'q'
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}