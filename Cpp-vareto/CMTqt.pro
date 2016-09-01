QT += core
QT -= gui

CONFIG += c++11

TARGET = CMTqt
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

INCLUDEPATH += /usr/local/include/opencv
LIBS += -L/usr/local/lib -lopencv_adas -lopencv_bgsegm -lopencv_bioinspired -lopencv_core -lopencv_datasets -lopencv_face -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_latentsvm -lopencv_line_descriptor -lopencv_ml -lopencv_objdetect -lopencv_optflow -lopencv_photo -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_surface_matching -lopencv_text -lopencv_tracking -lopencv_video -lopencv_videoio -lopencv_videostab -lopencv_xfeatures2d -lopencv_ximgproc -lopencv_xobjdetect -lopencv_xphoto

SOURCES += main.cpp \
    cmt.cpp \
    common.cpp \
    consensus.cpp \
    fusion.cpp \
    matcher.cpp \
    tracker.cpp \
    fastcluster.cpp \
    gui.cpp

HEADERS += \
    cmt.h \
    common.h \
    consensus.h \
    fusion.h \
    matcher.h \
    tracker.h \
    fastcluster.h \
    gui.h
