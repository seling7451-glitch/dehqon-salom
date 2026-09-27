---
title: Dehqon AI Model Server
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Dehqon AI — model server

Bu Space `POST /predict` endpointi orqali o'simlik barg rasmini qabul qilib,
o'simlik va kasallik turini, davolash va oldini olish tavsiyalarini qaytaradi.

So'rov formati (mobil ilovadagi kabi, o'zgarmagan):

```
POST /predict
Content-Type: image/jpeg (yoki rasm mime turi)
Body: rasm baytlari (raw)
Header (ixtiyoriy): X-Plant-Type: Tomato
```
