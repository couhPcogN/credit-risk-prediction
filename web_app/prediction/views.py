from django.http import FileResponse, Http404
from django.shortcuts import render

from .ml_service import DEFAULT_INPUTS, FORM_OPTIONS, get_credit_risk_service, resolve_report_path


def index(request):
    result = None
    predictions_results = None
    error = None

    try:
        service = get_credit_risk_service()
        inputs = service.display_inputs(request.POST if request.method == "POST" else None)

        if request.method == "POST":
            try:
                predictions_results = service.predict_all(request.POST)
                result = service.predict(request.POST)
                inputs = result.inputs
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                error = f"Prediction could not be processed: {exc}"

        context = {
            "inputs": inputs,
            "defaults": DEFAULT_INPUTS,
            "options": FORM_OPTIONS,
            "result": result,
            "predictions_results": predictions_results,
            "error": error,
            "summary": service.summary,
            "class_distribution": service.class_distribution(),
            "dataset_profile": service.dataset_profile,
            "reports": service.reports(),
            "service_ready": True,
        }
    except Exception as exc:
        context = {
            "inputs": DEFAULT_INPUTS,
            "defaults": DEFAULT_INPUTS,
            "options": FORM_OPTIONS,
            "result": None,
            "predictions_results": None,
            "error": f"Models could not be initialized: {exc}",
            "summary": None,
            "class_distribution": [],
            "dataset_profile": {"grades": [], "intents": []},
            "reports": [],
            "service_ready": False,
        }

    return render(request, "index.html", context)


def report_image(request, filename):
    path = resolve_report_path(filename)
    if path is None:
        raise Http404("Report image not found")
    return FileResponse(path.open("rb"), content_type="image/png")
