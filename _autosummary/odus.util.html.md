# odus.util

General utilities: image conversion and multi-image file writing.

Converts matplotlib figures / arrays / PIL images to `PIL.Image` and writes a
sequence of them to a single file (e.g. a multi-page pdf).

### Functions

| [`func_to_get_pil_images_like_this_one`](#odus.util.func_to_get_pil_images_like_this_one)(x, \*\*kwargs)   | Get a function that will convert images, specified as the example input x, into PIL.Image types   |
|--------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| [`write_images`](#odus.util.write_images)(images[, fp, pil_write_format, ...])     | Write images or figs into pdf pages (one image per page)                                          |

### odus.util.func_to_get_pil_images_like_this_one(x, \*\*kwargs)

Get a function that will convert images, specified as the example input x, into PIL.Image types

### odus.util.write_images(images, fp='test.pdf', pil_write_format=None, to_pil_image_kwargs=None, \*\*pil_save_kwargs)

Write images or figs into pdf pages (one image per page)

* **Parameters:**
  * **images** – iterable of “images” (should be a matplotlib figure, a PIL.Image,
    or a type that PIL.Image.open can open”)
  * **fp** – The filepath, path.Path, or file object to dump the images into
  * **\*\*params** – # extra parameters that are passed on to the save function

### Returns
