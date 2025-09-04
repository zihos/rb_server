# rb_server

sam vit-h pt [download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
pt파일 다운받아서 `/weights`에 넣기

### conda setting
```
conda create -n fastapi python=3.10 -y
conda activate fastapi
pip install fastapi uvicorn[standard]
pip install fastapi uvicorn[standard] pillow python-multipart

#sam + yolo
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics
```

### sever start
```
# server activate
uvicorn main:app --reload
```

### QT .pro
```
QT += concurrent
QT += network
LIBS += -lcurl
```

### mainwindow.h
```
#include "image_inference_client.h"
```

### mainwindow.cpp
```
void MainWindow::on_BTN_3D_RECONSTRUCTION_PAST_DATA_clicked() {
    QString selectedDate = selectPastDataDate();
    if (selectedDate.isEmpty()) return;


    QString picDir = QSettings("config.ini", QSettings::IniFormat).value("Paths/PicDir", "./pic").toString();
    QString dataPath = QString("%1/trigger/%2").arg(picDir).arg(selectedDate);
    const std::string imageFilePath = (dataPath + "/pattern_19.bmp").toStdString();


    bool useGPU = config.useGPU;


    // 2. 비동기: AI 서버 요청
    QFuture<cv::Mat> futureSAM = QtConcurrent::run([=]() -> cv::Mat {
        QElapsedTimer timer;
        timer.start();
        qDebug() << "[SAM] start for preparing masked image" << timer.elapsed() <<"ms";


        ImageInferenceClient client("http://127.0.0.1:8000");
        client.sendImage(imageFilePath);


        QJsonObject sam = client.fetchSAMResults();
        if (!sam.contains("combined_mask_b64")) return {};


        QByteArray img_data = QByteArray::fromBase64(sam["combined_mask_b64"].toString().toUtf8());
        std::vector<uchar> data(img_data.begin(), img_data.end());
        cv::Mat decoded_img = cv::imdecode(data, cv::IMREAD_GRAYSCALE);
        qDebug() << "[SAM] ready for masked image" << timer.elapsed() <<"ms";
        return decoded_img;
    });


    // 1. 비동기: 3D Reconstruction
    QFuture<void> future3D = QtConcurrent::run([=]() {
        if (useGPU)
            mReconstruct3D.Perform3DReconstructionGPU(dataPath);
        else
            mReconstruct3D.Perform3DReconstruction(dataPath);
    });




    // 3. 완료 감지
    QFutureWatcher<cv::Mat>* watcher = new QFutureWatcher<cv::Mat>(this);
    connect(watcher, &QFutureWatcher<cv::Mat>::finished, this, [=]() {
        watcher->deleteLater();  // clean up
        QElapsedTimer timer;
        timer.start();


        // futureSAM 결과 받기
        m2DMaskImage = futureSAM.result();


        if (m2DMaskImage.empty()) {
            QMessageBox::warning(this, "Error", "Failed to decode mask image");
            return;
        }


        // threshold + flag
        cv::threshold(m2DMaskImage, m2DMaskImage, 127, 255, cv::THRESH_BINARY);
        mHas2DMask = true;


        qDebug() << "[SAM] Masked Image Decoded ";




        // 4. 이후는 동기적으로 처리
        display2DMaskImage();
        apply2DMaskToPointCloud();
        qDebug() << "[SAM] Masked Image Applied."<< timer.elapsed() <<"ms";
    });


    // 4. 시작
    watcher->setFuture(futureSAM);
}
```
