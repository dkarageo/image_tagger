# Image Tagger
#### A REST API for predicting ImageNet labels of images.

## Description
Image Tagger is REST API for tagging images with ImageNet-1K labels.
It wraps around an _EfficientNetB3_ model trained on _ILSVRC_ dataset,
exposing its functionality through a simple endpoint. The API was built using
_FastAPI_ python framework. It utilizes _Tensorflow_ for executing deep 
learning models.  

## Quick Start
Image Tagger is available at PyPI and can be installed as following:

1. `pip install image_tagger`

In order to start the server use:

2. `py -m image_tagger [host=0.0.0.0] [port=80]`

Now the server should be up and running. 

Get the url of an image you like and perform a GET request to the API like:

3. `GET localhost/predictions?url=https://example.com/path/to/image.png`

You will see a nice JSON response containing the label for your image. 
You can perform a GET request either by navigating to the above location
with your browser, or by using a requests tool like _Postman_.

4. Use your favorite browser and navigate to `localhost/docs`.

You will see a nice documentation page for all the endpoints
exposed by this API, along with the ability for interactively
calling them. 

5. Navigate to `localhost/redoc`.

You will see another version of the documentation. This one is for
people who prefer the style of _Redoc_ docs, instead of the _Swagger UI_
based ones.

## Endpoints

Currently, two endpoints are exposed by the API. The one for performing
predictions and the other for obtaining versioning info about the live API.

### GET `/predictions`
##### Predicts the ImageNet-1K labels of an image.

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

#### Parameters

- **url**: The url of the image file. It can be any https or http url pointing
   to an image file. For example: 'https://example.com/path/to/image.jpg'
- **top** [optional]: The number of top ImageNet labels to be returned. The number
   of returned labels cannot exceed the number of available classes
   (top<=1000).
- **minconf** [optional]: The minimum confidence score a label should have in order to be
   returned. It should be in the range `[0, 1]`.

#### Responses

Successful calls return a JSON containing the following attributes:

* __url__: The same url passed into the request.
* __predictions__: An array containing the predicted label objects in descending order.
   Each object contains the following attributes:
   - __label__: The predicted ImageNet-1K label.
   - __confidence__: The confidence score of the label.

###### Example
```
{
    "url": "https://example.com/path/to/image.jpg",
    "predictions": [
        {"label": "dog", "confidence": 0.78},
        {"label": "wolf", "confidence": 0.06},
        {"label": "tiger", "confidence": 0.03},]
}
```

##### Error 400: InvalidImage

In case the `url` points to a resource that is not a supported image format,
response code is set to `400` and the following JSON is returned:
```
{"error": {
    "type": "InvalidImage",
    "detail": "Url is not pointing to a valid image format."
}}
```

##### Error 404: UnreachableURL

If the GET request on `url` fails, then response code is set to `404` and the
   following JSON is returned:
```
{"error": {
    "type": "UnreachableURL",
    "detail": "Url cannot be reached."
}}
```

##### Error 422: ValidationError

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

### GET `/`
##### Returns general info about image tagger API.

This endpoint serves both as a placeholder for root path and as a way in order
to retrieve versioning info.

#### Responses

Calls return a JSON containing the following attributes:

* __description__: A short textual description of the API.
* __version__: The version of the API in the format of semantic versioning
   (e.g. 1.0.0). For more info see: https://semver.org/
* __revision__: An integer indicating the current revision of the API.
   Each newer version of the API is guaranteed to have a greater revision
   number.
* __author__: API author's name.
* __github_url__: Link to the github repository containing the code of this API.

## Documentation

Except from the REST endpoints described above, Image Tagger also provides a live version
of their documentation.

### `/docs`: Swagger UI based documentation.

### `/redoc`: Redoc based documentation.

## License:
This project is licensed under Apache License 2.0. A copy of this license is contained in current 
project under `LICENSE` file. It applies to all files in this project whether or not it is stated in them.

Copyright 2022 | Dimitrios S. Karageorgiou
