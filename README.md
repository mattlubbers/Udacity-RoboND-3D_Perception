## **Project: Perception Pick & Place**
---
### **Building a Perception Pipeline:**
#### 1.1) Filtering and RANSAC Plane Fitting
Statistical Outlier Filtering:
```python
    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    #Statistical Outlier Filter
    outlierFilt = cloud.make_statistical_outlier_filter()
    outlierFilt.set_mean_k(3)
    x = 0.00001
    outlierFilt.set_std_dev_mul_thresh(x)
    cloud = outlierFilt.filter()
```
##### Voxel Downsampling and Pass-Through Filter
Downsample the cloud density and filter out regions of the environment that do not contain objects of interest:
```python
    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.3
    axis_max = 2.1
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
```
##### RANSAC Plane Segmentation
Filter out the table to focus processing on the objects:
```python
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)
```
#### 1.2) Clustering for Segmentation  
##### Euclidean Clustering
Create the white cloud:
```python
white_cloud = XYZRGB_to_XYZ(cloud_objects)
tree = white_cloud.make_kdtree()
```
##### Cluster-Mask Point Cloud to Visualize each Cluster
Set clustering parameters (Tolerance, min/max Size):
```python
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    
    #Assign color to each object
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
```
##### White Cloud to RGB
Iterate through white point cloud to construct RGB color point cloud clusters:
```python
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
```
##### PCL data to ROS and publish messages
Convert Point Cloud to ROS Messages and Publish:
```python
    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)
```
#### 1.3) Features Extracted
Compute the Feature Vector:
```python
    chists = compute_color_histograms(ros_cluster, using_hsv=True)
    normals = get_normals(ros_cluster)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))
```
Predict and Append Object Label:
```python
    prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
    label = encoder.inverse_transform(prediction)[0]
    detected_objects_labels.append(label)
```
Publish Label into RViz:
```python
    label_pos = list(white_cloud[pts_list[0]])
    label_pos[2] += .4
    object_markers_pub.publish(make_label(label,label_pos, index))
```
Add Object to Object List:
```python
    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []
```
##### 1.4) Object Detection Training
##### Capture Ojbects in Test World
The objects that we are attempting to detect, classify, and label will need to be collected, and trained. Each object will be rotated in an isolated environment to collect a point cloud in multiple orientations of the object. This will be repeated until each of the objects has sufficient data:
```python
if __name__ == '__main__':
    rospy.init_node('capture_node')

    models = [\
       'beer',
       'bowl',
       'create',
       'disk_part',
       'hammer',
       'plastic_cup',
       'soda_can']

    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        spawn_model(model_name)

        for i in range(5):
            # make five attempts to get a valid a point cloud then give up
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 5:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=False)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])

        delete_model()


    pickle.dump(labeled_features, open('training_set.sav', 'wb'))
```
##### SVM Training
The dataset for each object has now been collected, and the Support Vector Machine (SVM) will now be trained to provide a high accuracy object label by characterizing the feature vector. The Normalized Confusion Matrix can be seen below:
![Confusion Matrix](/Assets/Normalized_Confusion_Matrix.png)

### **Pick and Place**
#### 2.1) Create tabletop setups (`test*.world`)
For each new test world, modify the launch file to load correct shopping items

#### 2.2) Perform Object Recognition
Use the perception pipeline and trained model to detect and label objects on the table for each test world
#### 2.3) Read in Respective Pick Pist (`pick_list_*.yaml`)
Load pick list for given test world to determine which objects should be selected, as well as the bin they must be placed
#### 2.4) Construct Messages for `PickPlace` request and Output to a `.yaml` file
Output the object label, location, and destination bin


And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

#### What worked!


#### Struggles


#### Next Steps and Optimizations
One aspect of this project that was determined to be out of scope was the actuation of the arm to physically grasp the object and move to the representative bin. The next steps would be to use the existing perception pipeline to complete the task of pick and place for end to end detection through placement. 

As discussed, the manual configuration of the test world selection, pick list items, capture, and training was a bit cumbersum and time intensive. The first optimization I would make is to make this automated by either an Input/Output command from the user, or a "run all" script.

This implementation lacks significant robustness testing, both for the object detection and filtering. Due to the significant differences in the object shape or color, distinction between objects was relatively straight forward and yielded adequate training accuracy numbers, however these current methods and parameters may run into lower accuracy performance with a more challenging object list. Similarly for the statistical outlier filtering and RANSAC filter for the Table, these parameters were manually calibrated for the specific environment, and may cause inaccuracies with a different test world.



