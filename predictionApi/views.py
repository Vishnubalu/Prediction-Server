import json

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from predictionApi import ML_model


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

        res = ML_model.prediction(data)

        # return render('index.html', template_name='index.html')
        return Response(json.dumps(res), content_type="application/json")

    except ValueError as e:
        print(e.args[0], status.HTTP_400_BAD_REQUEST)
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)