import urllib.error
from typing import Optional

import PIL
from fastapi import FastAPI, Query, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import AnyHttpUrl

from .image_predictions import predict_imagenet_labels
from .response_models import HttpErrorResponse, ValidationErrorResponse, PredictionResponse, InfoResponse


__VERSION__ = "1.0.0"
__REVISION__ = 1
__DESCRIPTION__ = """
__A simple API for predicting ImageNet-1K labels of an image.__

Currently, image tagger wraps around an _EfficientNetB3_ model, trained on _ILSVRC_ dataset.

The main endpoint for performing predictions is `/predictions`.

Versioning info can be retrieved with a GET request to root `/` endpoint.  

Documentation for all supported endpoints is provided under:
* `/docs`: Swagger UI style documentation with interactive console.  
* `/redoc`: Redoc style documentation.
"""

app = FastAPI(
    title="Image Tagger API",
    description=__DESCRIPTION__,
    version=__VERSION__,
    contact={
        "name": "Dimitrios S. Karageorgiou",
        "url": "https://github.com/dkarageo/image_tagger",
        "email": "soulrain@outlook.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Static response to be returned be the root path.
info_response = {
        "description": __DESCRIPTION__,
        "version": __VERSION__,
        "revision": __REVISION__,
        "author": "Dimitrios S. Karageorgiou",
        "github_url": "https://github.com/dkarageo/image_tagger",
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Exception handler for validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({
            "error": {
                "type": "ValidationError",
                "detail": exc.errors()
            }
        })
    )


@app.get("/",
         response_model=InfoResponse,
         responses={
             status.HTTP_200_OK: {"content": {"application/json": {"example": info_response}}}},
         summary="Returns general info about the API.",
         tags=["info"])
async def info():
    """# Returns general info about image tagger API.

    This endpoint serves both as a placeholder for root path and as a way in order
    to retrieve versioning info.

    ## Responses

    Calls return a JSON containing the following attributes:

    * __description__: A short textual description of the API.
    * __version__: The version of the API in the format of semantic versioning
       (e.g. 1.0.0). For more info see: https://semver.org/
    * __revision__: An integer indicating the current revision of the API.
       Each newer version of the API is guaranteed to have a greater revision
       number.
    * __author__: API author's name.
    * __github_url__: Link to the github repository containing the code of this API.
    """
    return info_response


@app.get("/predictions",
         response_model=PredictionResponse,
         responses={
             status.HTTP_200_OK: {
                 "content": {"application/json": {"example": {
                     "url": "https://example.com/path/to/image.jpg",
                     "predictions": [
                         {"label": "dog", "confidence": 0.78},
                         {"label": "wolf", "confidence": 0.06},
                         {"label": "tiger", "confidence": 0.03},
                     ]}}}},
             status.HTTP_422_UNPROCESSABLE_ENTITY: {
                 "model": ValidationErrorResponse,
                 "description": "Invalid query parameters.",
                 "content": {"application/json": {"example": {
                     "error": {
                         "type": "ValidationError",
                         "detail": [{"loc": ["query", "url"],
                                     "message": "URL scheme not permitted",
                                     "type": "value_error.url.scheme"}]
                     }}}}
             },
             status.HTTP_400_BAD_REQUEST: {
                 "model": HttpErrorResponse,
                 "description": "Given url is not pointing to a valid image format.",
                 "content": {"application/json": {"example": {
                     "error": {
                         "type": "InvalidImage",
                         "detail": "Url is not pointing to a valid image format."
                     }}}}},
             status.HTTP_404_NOT_FOUND: {
                 "model": HttpErrorResponse,
                 "description": "Given url cannot be reached.",
                 "content": {"application/json": {"example": {
                     "error": {
                         "type": "UnreachableURL",
                         "detail": "Url cannot be reached."
                     }}}}}},
         summary="Predicts the ImageNet-1K labels of an image.",
         tags=["predictions"])
async def predict(url: AnyHttpUrl = Query(...), top: Optional[int] = Query(None, ge=1, le=1000),
                  minconf: Optional[float] = Query(None, qe=0.0, le=1.0)):
    """# Predicts the ImageNet-1K labels of an image.

    Returned labels can be filtered according to two parameters, `top` and `minconf`.
    The first parameter controls the number of top-confidence labels to be
    returned, sorted in descending order according to computed confidence. The
    second one, limits the returned labels to the ones whose confidence score is
    at least `minconf`. These two parameters can be combined, applying the restrictions
    of both, thus returning the most restrictive labels set.
    By default, if `top` and `minconf` are omitted, only the label with the highest
    confidence score is returned (the same as setting `top=1`).

    The image format can be any format supported be the Pillow library. For
    more info see: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

    Image retrieval is performed through a GET request on given url.

    ## Parameters

    - **url**: The url of the image file. It can be any https or http url pointing
       to an image file. For example: 'https://example.com/path/to/image.jpg'
    - **top** [optional]: The number of top ImageNet labels to be returned. The number
       of returned labels cannot exceed the number of available classes
       (top<=1000).
    - **minconf** [optional]: The minimum confidence score a label should have in order to be
       returned. It should be in the range `[0, 1]`.

    ## Responses

    Successful calls return a JSON containing the following attributes:

    * __url__: The same url passed into the request.
    * __predictions__: An array containing the predicted label objects in descending order.
       Each object contains the following attributes:
       - __label__: The predicted ImageNet-1K label.
       - __confidence__: The confidence score of the label.

    ### Example
    ```
    {
        "url": "https://example.com/path/to/image.jpg",
        "predictions": [
            {"label": "dog", "confidence": 0.78},
            {"label": "wolf", "confidence": 0.06},
            {"label": "tiger", "confidence": 0.03},]
    }
    ```

    ### Error 400: InvalidImage

    In case the `url` points to a resource that is not a supported image format,
    response code is set to `400` and the following JSON is returned:
    ```
    {"error": {
        "type": "InvalidImage",
        "detail": "Url is not pointing to a valid image format."
    }}
    ```

    ### Error 404: UnreachableURL

    If the GET request on `url` fails, then response code is set to `404` and the
       following JSON is returned:
    ```
    {"error": {
        "type": "UnreachableURL",
        "detail": "Url cannot be reached."
    }}
    ```

    ### Error 422: ValidationError

    When the parameters of the request are invalid, response code is set to `422` and
       a JSON describing the cause of the error is returned. An example response is
       presented below:
    ```
    {"error": {
        "type": "ValidationError",
        "detail": [{"loc": ["query", "url"],
                    "message": "URL scheme not permitted",
                    "type": "value_error.url.scheme"}]
    }}
    ```

    The only part of the response that is changing for different validation errors
    is the content of `detail` attribute. It is an array containing all validation
    errors detected. The detail object of each detected violation contains the
    following attributes:

    * __loc__:  An array describing the location of the error. For example
       `["query", "url"]` means that the value of query parameter `url` is invalid.
    * __message__: An informal message describing the cause of the error.
    * __type__: A formal identifier of the error.
    """
    try:
        predictions = predict_imagenet_labels(str(url), top, minconf)
    except (urllib.error.HTTPError, urllib.error.URLError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=jsonable_encoder({
                "error": {
                    "type": "UnreachableURL",
                    "detail": "Url cannot be reached."
                }
            })
        )
    except PIL.UnidentifiedImageError:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=jsonable_encoder({
                "error": {
                    "type": "InvalidImage",
                    "detail": "Url is not pointing to a valid image format."
                }
            })
        )

    response = {
        "url": url,
        "predictions": predictions
    }
    return response
