import json
import django
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import os
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from predictionApi import ML_model
import time

@api_view(['GET'])
def liverDiseaseSymptoms(request):
    data = {
        "symptoms" : ML_model.getcols()
    }
    return Response(data)

@api_view(['POST'])
def liverDiseasePrediction(request):
    try:
        data = json.load(request)
        print(data)
        res = ML_model.prediction(data)
        return Response(json.dumps({"messege":res}), content_type="application/json")

    except ValueError as e:
        print(e.args[0], status.HTTP_400_BAD_REQUEST)
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(["POST"])
def uploadFile(request):
    try:
        data = json.load(request)
        # print(data["data"])
        #file = data['filename']
        df = pd.DataFrame(data=data["data"])
        df.dropna(inplace=True)
        df_copy = df.copy()
        print(df.head())
        ans = ML_model.predictFromCSV(df_copy)
        if ans[0] == 'incorrect':
            raise ValueError
        df["result"] = ans
        encode = {1:"have disease", 2:"not have disease"}
        df["result"] = df["result"].map(encode)
        path = os.getcwd() + r'\data\results.csv'
        df.to_csv(path,index = False, header=True)
        return Response(json.dumps({"result":"ok", "created" : 0, "open" : 0}), content_type="application/json")

    except ValueError as e:

        print("error returning")
        return Response(json.dumps({"result":"symptoms are not matching, PLEASE KINDLY FOLLOW THE SAMPLE FORMAT",
                                    "created" : 1, "open" : 1}), content_type="application/json")

@csrf_exempt
def downloadsample(request):
    path = os.getcwd()
    data = open(os.path.join(path, 'data/sample.csv'),'r').read()
    resp = django.http.HttpResponse(data, content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename = sample.csv'
    return resp

@csrf_exempt
def downloadResult(request):

    path = os.getcwd() + r'\data\results.csv'
    data = open(os.path.join(path), 'r').read()
    resp = django.http.HttpResponse(data, content_type='text/csv')
    resp['Content-Disposition'] = 'attachment; filename = "{}"'.format(str(time.ctime())+"_Result.csv")
    return resp