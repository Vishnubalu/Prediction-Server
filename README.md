# Prediction-Server
This repository contains API to predict liver disease


```
/POST
/predictdisease

RequestBody
 {
    "symptoms": {
        "Age":"12",
        "Gender":"1",
        "Total_Bilirubin":"12",
        "Direct_Bilirubin":"1.2",
        "Alkaline_Phosphotase":"65",
        "Alamine_Aminotransferase":"2",
        "Aspartate_Aminotransferase":"98",
        "Total_Protiens":"12",
        "Albumin":"23",
        "Albumin_and_Globulin_Ratio":"23"
    }
}

ResponseBody
   {"messege": "not have disease"}


```
