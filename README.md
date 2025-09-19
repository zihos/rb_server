# rb_server
pth파일 다운받아서 `/weights`에 넣기

sam vit-h pt [download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

sam vit-b pt [download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

**mobile sam**

`main.py`에서 checkpoint경로 지정
```
SAM_CKPT     = os.environ.get("SAM_CKPT", "./weights/mobile_sam.pt")
SAM_ONNX     = os.environ.get("SAM_ONNX", "./weights/mobile_sam_onnx_vit_t.onnx")
SAM_TYPE     = os.environ.get("SAM_TYPE", "vit_t")  # vit_h | vit_l | vit_b | vit_t
```



### conda setting
```
conda create -n fastapi python=3.10 -y
conda activate fastapi
pip install fastapi uvicorn[standard] pillow python-multipart

#sam + yolo (rb_server 밖에서 sam 설치)
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics
```

### sever start
```
# import tensorrt
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3.10/dist-packages
# server activate
uvicorn dice_main:app --reload
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


    // Get JIG transform matrix if active
    cv::Mat jigTransform;
    if (m_jigTransformActive && !m_jigTransform.empty()) {
        jigTransform = m_jigTransform;
        qDebug() << "JIG transform is active, will apply to 3D reconstruction";
    }

    // 1. 비동기: 3D Reconstruction
    QFuture<void> future3D = QtConcurrent::run([=]() {
        if (useGPU)
            mReconstruct3D.Perform3DReconstructionGPU(dataPath, jigTransform);
        else
            mReconstruct3D.Perform3DReconstruction(dataPath, jigTransform);
    });

    // 2. 비동기: SAM 서버 요청
    QFuture<cv::Mat> futureSAM = QtConcurrent::run([=]() -> cv::Mat {
        QElapsedTimer timer;
        timer.start();
        qDebug() << "[SAM] start for preparing masked image" << timer.elapsed() << "ms";

        // 예: 서버 호출 · 결과 수신
        ImageInferenceClient client("http://127.0.0.1:8000");
//        client.sendImage("/mnt/ssd/pattern_19/pattern_19_5.bmp");
        client.sendImage(imageFilePath);
        QJsonObject sam = client.fetchSAMResults();

        qDebug() << "[SAM] receive results, num of generated masks:" << sam["num_instances"].toInt();

        // 2-1) 마스크 없음 → 검은 이미지 반환 (원본 크기와 동일)
        if (!sam.contains("num_instances") || sam["num_instances"].toInt() == 0) {
            cv::Mat orig = cv::imread(imageFilePath);
            if (orig.empty()) {
                // 원본도 못 읽으면 안전한 기본 크기
                return cv::Mat(480, 640, CV_8UC1, cv::Scalar(0));
            }
            qDebug() << "[SAM] No mask instances, returning black image ->"
                     << orig.rows << "x" << orig.cols;
            return cv::Mat(orig.rows, orig.cols, CV_8UC1, cv::Scalar(0));
        }

        // 2-2) 마스크 있음 → base64 디코드
        if (sam.contains("combined_mask_b64")) {
            QByteArray img_b64 = sam["combined_mask_b64"].toString().toUtf8();
            QByteArray raw = QByteArray::fromBase64(img_b64);

            std::vector<uchar> buf(raw.begin(), raw.end());
            cv::Mat decoded = cv::imdecode(buf, cv::IMREAD_GRAYSCALE);
            if (decoded.empty()) {
                // 디코드 실패 → 검은 이미지 폴백
                cv::Mat orig = cv::imread(imageFilePath);
                if (orig.empty()) return cv::Mat(480, 640, CV_8UC1, cv::Scalar(0));
                qDebug() << "[SAM] decode failed, returning black image";
                return cv::Mat(orig.rows, orig.cols, CV_8UC1, cv::Scalar(0));
            }
            qDebug() << "[SAM] ready for masked image" << timer.elapsed() << "ms";
            return decoded;
        }

        // 2-3) 필드 자체가 없으면 폴백
        cv::Mat orig = cv::imread(imageFilePath);
        if (orig.empty()) return cv::Mat(480, 640, CV_8UC1, cv::Scalar(0));
        qDebug() << "[SAM] mask field missing, returning black image";
        return cv::Mat(orig.rows, orig.cols, CV_8UC1, cv::Scalar(0));
    });

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


    // display 2d mask image on the preview window
    display2DMaskImage();

    // 3. 완료 감지: watcher 두 개 + 카운터
    auto *watcherSAM = new QFutureWatcher<cv::Mat>(this);
    auto *watcher3D  = new QFutureWatcher<void>(this);

    // 남은 작업 개수. shared_ptr로 안전하게 관리
    auto remaining = std::make_shared<int>(2);

    auto tryProceed = [=]() mutable {
        (*remaining)--;
        if (*remaining > 0) return;  // 아직 하나 남음

        QElapsedTimer timer; timer.start();

        // 4. 이후는 동기 처리 (UI 갱신 등)
        apply2DMaskToPointCloud();
        qDebug() << "[SAM] Masked Image Applied." << timer.elapsed() << "ms";

        // 정리
        watcherSAM->deleteLater();
        watcher3D->deleteLater();
    };

    // 4. 시그널 연결 + 감시 시작
    connect(watcherSAM, &QFutureWatcher<cv::Mat>::finished, this, tryProceed);
    connect(watcher3D,  &QFutureWatcher<void>::finished,   this, tryProceed);

    watcherSAM->setFuture(futureSAM);
    watcher3D->setFuture(future3D);
}
```
