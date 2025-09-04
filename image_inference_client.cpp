#include "image_inference_client.h"

#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QHttpMultiPart>
#include <QHttpPart>
#include <QFile>
#include <QEventLoop>
#include <QJsonDocument>
#include <QBuffer>
#include <QByteArray>
#include <QImage>
#include <QImageReader>
#include <QDebug>
#include <QJsonArray>
#include <QJsonObject>

ImageInferenceClient::ImageInferenceClient(const std::string& server_url)
    : serverUrl(server_url) {}

bool ImageInferenceClient::sendImage(const std::string& image_path) {
    QByteArray response = sendPostImage("/image", image_path);
    return !response.isEmpty();
}

QJsonObject ImageInferenceClient::fetchYoloResult() {
    return sendGetRequest("/yolo");
}

QJsonObject ImageInferenceClient::fetchSAMResults() {
    return sendGetRequest("/masks");
}

std::vector<QImage> ImageInferenceClient::fetchMasks() {
    QJsonObject response = sendGetRequest("/masks");
    std::vector<QImage> mask_images;

    if (!response.contains("masks_bmp_b64")) return mask_images;


    // single masked images
    QJsonArray masks_array = response["masks_bmp_b64"].toArray();
    for (const QJsonValue &val : masks_array) {
        QByteArray base64_data = val.toString().toUtf8();
        QByteArray img_data = QByteArray::fromBase64(base64_data);

        qDebug() << ">> decoded base64 length:" << img_data.size();
        qDebug() << ">> img_data type:" << typeid(img_data).name();

        QImage img;
        img.loadFromData(img_data);

        qDebug() << ">> img.format:" << img.format();
        if (!img.isNull()) {
            mask_images.push_back(img);

            QLabel* label = new QLabel;
            label -> setPixmap(QPixmap::fromImage(img).scaled(img.size()/2, Qt::KeepAspectRatio));
            label -> setWindowTitle("Decoded Mask Image");
            label -> resize(img.size()/2);
            label -> show();
        } else{
            qDebug() << "[ERROR] Failed to load image";
        }
    }
    //combined masked image
    QString combined_b64 = response["combined_mask_b64"].toString();
    QByteArray combined_data = QByteArray::fromBase64(combined_b64.toUtf8());
    QImage combined_img;
    combined_img.loadFromData(combined_data);

//    if (!combined_img.isNull()){
//        QLabel* label = new QLabel;
//        label -> setPixmap(QPixmap::fromImage(combined_img).scaled(combined_img.size()/2, Qt::KeepAspectRatio));
//        label -> setWindowTitle("Decoded Combined Mask Image");
//        label -> resize(combined_img.size()/2);
//        label -> show();
//    }

    return {combined_img};
}

QByteArray ImageInferenceClient::sendPostImage(const std::string& endpoint, const std::string& image_path) {
    QUrl url(QString::fromStdString(serverUrl + endpoint));
    QNetworkRequest request(url);
    QNetworkAccessManager manager;

    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);
    QHttpPart imagePart;

    QFile *file = new QFile(QString::fromStdString(image_path));
    if (!file->open(QIODevice::ReadOnly)) {
        qWarning() << "Cannot open file:" << image_path.c_str();
        delete file;
        return {};
    }

    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader,
                        QVariant("form-data; name=\"file\"; filename=\"" + file->fileName() + "\""));
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    imagePart.setBodyDevice(file);
    file->setParent(multiPart); // 자동 삭제

    multiPart->append(imagePart);

    QNetworkReply *reply = manager.post(request, multiPart);
    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    QByteArray response_data = reply->readAll();
    reply->deleteLater();
    return response_data;
}

QJsonObject ImageInferenceClient::sendGetRequest(const std::string& endpoint) {
    QUrl url(QString::fromStdString(serverUrl + endpoint));
    QNetworkRequest request(url);
    QNetworkAccessManager manager;

    QNetworkReply *reply = manager.get(request);
    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    QByteArray response = reply->readAll();
    reply->deleteLater();

    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(response, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        qWarning() << "Failed to parse JSON from" << endpoint.c_str();
        return {};
    }

    return doc.object();
}
