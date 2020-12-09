# Deep Image clustering of Mars Orbital Survey (DeIMOS)

#### Lee Ding, Joe Han, George Hu

Create a folder within this called "data", and download the zip in there. Use the sample_images.py script to sample images from the zipfile.

### Using the VM
You can just use the browser console from the GCloud page by clicking SSH: https://console.cloud.google.com/compute/
Note: visualization will not work here as it is terminal-only. 

Guide on getting a GUI: https://medium.com/google-cloud/linux-gui-on-the-google-cloud-platform-800719ab27c5
XFCE is already installed, so only worry about the VNC server; alternatively, same some time and effort
by using Chrome Remote Desktop: https://cloud.google.com/solutions/chrome-desktop-remote-on-compute-engine

Other notes: Code files are in /joe_qiyu_han/DeIMOS. Use 'sudo su' to get root permissions. 

### TODO: (I'm putting this here since it's surprisingly convenient and I want to make my github commit history look better.)

- To run the current abomination (okay it's not an abomination but it does suck.), just download a bunch of data, and run `python3 test.py`
- Parameters to tune:
    - In `test.py`
        - `learning_rate` in the `tsne_params` dict (ranges from 10 to 1000 and I don't really understand it tbh)
        - Other stuff for tsne parameters, check the sklearn for it.
        - `n_clusters` in the model initialization. Since kmeans kind of failed (perhaps I can still salvage it with resnet > mobilenet? not sure), I don't know how many clusters it should be.
        - `n_epochs` self-explanatory. Don't think this matters too much rn but idk.
    - In `model.py`
        - `self.lr` self-explanatory
        - `self.u_coeff` and `self.l_coeff`; these determine how quickly we want to include more training data (the paradigm is you select training data based-off similarity in terms of wanting very similar or very dissimilar training data, and as the model trains, this becomes less selective). `u_coeff` governs how quickly we want to include similar data, and `l_coeff` governs how quickly we want to include dissimilar data. I think we can definitely try increasing `l_coeff` and decreasing `u_coeff`, since rn it's making everything too similar
        - Conv layer dimensionality. Maybe you can try increasing the number of filters for convolutions? Currently it's an approximation of 1/4 the number of filters as VGG, so maybe we need something better. I wouldn't advise changing the other stuff since the shape get's all messed up, but you can try if you want.
        - Dense layer dimensionality. Can try messing around with this, but don't change dimensionality of output layer.
        - `0.95` in `upper_bound` and `0.455` in `lower_bound`. These are related to u_coeff and l_coeff as described earlier. These values are the initial boundaries for similarity and dissimilarity.
