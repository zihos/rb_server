#ifndef IMAGEINFERENCECLIENT_H
#define IMAGEINFERENCECLIENT_H

// temoporal (zio)
#include <QNetworkAccessManager>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QUrl>
#include <QUrlQuery>
#include <QByteArray>
#include <QString>
#include <QHttpPart>
#include <QHttpMultiPart>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <iostream>
#include <fstream>
#include <sstream>
#include <curl/curl.h>
#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>
#include <string>
#include <vector>
#include <QJsonObject>
#include <QImage>
#include <QLabel>

class ImageInferenceClient
{
public:
    explicit ImageInferenceClient(const std::string& server_url);

    // 1. upload image POST
    bool sendImage(const std::string& imagePath);

    // 2. yolo results GET
    QJsonObject fetchYoloResult();

    // 3. masked images GET
    std::vector<QImage> fetchMasks();
    QJsonObject fetchSAMResults() ;
private:
    std::string serverUrl;

    QJsonObject sendGetRequest(const std::string& endpoint);
    QByteArray sendPostImage(const std::string& endpoint, const std::string& image_path);
};

#endif // IMAGEINFERENCECLIENT_H
