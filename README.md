# Deep Image clustering of Mars Orbital Survey (DeIMOS)

#### Lee Ding, Joe Han, George Hu

Create a folder within this called "data", and download the zip in there. Use the sample_images.py script to sample images from the zipfile.



### TODO: (I'm putting this here since it's surprisingly convenient and I want to make my github commit history look better.)

- Modify `sample_images.py` so that you download labeled and unlabled images into separate subfolders.
- Create helper functions that will return `tf.data.Dataset` type for labeled images and unlabeled images (using the image file paths in the subfolders and labels in the text file). These should contain tensors of shape `(227, 227, 1)` of datatype `tf.float32` for each image. Also make sure to include the label in the labeled images dataset.
- You probably need to reference `tf.io` and `tf.image` for some of these things, along with [this](https://www.tensorflow.org/guide/data#consuming_sets_of_files). 
