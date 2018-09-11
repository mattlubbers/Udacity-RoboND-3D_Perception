#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)
    return

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    #Statistical Outlier Filter
    outlierFilt = cloud.make_statistical_outlier_filter()
    outlierFilt.set_mean_k(3)
    x = 0.00001
    outlierFilt.set_std_dev_mul_thresh(x)
    cloud = outlierFilt.filter()

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

    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()

    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.015)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    
    #Assign color to each object
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)

    # Publish ROS messages
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        #labeled_features.append([feature, model_name])
        
        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
   
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    if len(detected_objects) > 0:    
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    else: 
        rospy.logwarn('No objects detected')
    	try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    return

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    test_scene_num = Int32()
    arm_name = String()
    object_name = String()
    pick_pose = Pose()
    place_pose = Pose()
    dict_list = []

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    #print(object_list_param)
    dropbox_param = rospy.get_param('/dropbox')

    test_scene_num.data = 2

    # Rotate PR2 in place to capture side tables for the collision map
    #pr2_base_mover_pub.publish(1.57)
    #print('moving left')
    #rospy.sleep(5.0)
    #pr2_base_mover_pub.publish(-1.57)
    #print('moving right')
    #rospy.sleep(5.0)    
    #pr2_base_mover_pub.publish(0)
    #rospy.sleep(5.0)

    # Parse parameters into individual variables
    labels = []
    centroids = []
    for object in object_list:
        labels.append(object.label)
        points_arr = ros_to_pcl(object.cloud).to_array()
        centroids.append(np.mean(points_arr, axis=0)[:3])
	#print(object.label, points_arr)
	#print(object.label)
    # Loop through the pick list
    #print(len(object_list_param))
    for i in range(0, len(object_list_param)):    
        object_name.data = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        # Get the PointCloud for a given object and obtain it's centroid
	try:
	    index = labels.index(object_name.data)
	except ValueError:
	    print('Object not detected')
	    continue

        pick_pose.position.x = np.asscalar(centroids[index][0])
	pick_pose.position.y = np.asscalar(centroids[index][1])
	#print(index, pick_pose.position.y)
	pick_pose.position.z = np.asscalar(centroids[index][2])

        # Create 'place_pose' for the object		
	for j in range(0, len(dropbox_param)):
	    if object_group == dropbox_param[j]['group']:	    
		place_pose.position.x = dropbox_param[j]['position'][0]
		place_pose.position.y = dropbox_param[j]['position'][1]
		place_pose.position.z = dropbox_param[j]['position'][2]

        # Assign the arm to be used for pick_place
	if object_group == 'green':
	    arm_name.data = 'right'
	elif object_group == 'red':
	    arm_name.data = 'left'

        # Create a list of dictionaries 	
	yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
	#print(yaml_dict)
	print(object_name, pick_pose.position.y)	
	dict_list.append(yaml_dict)

        # Place item (disabled)
#        rospy.wait_for_service('pick_place_routine')
#        try:
#            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Message variables to be sent as a service request
#            resp = pick_place_routine(test_scene_num,object_name,arm_name,pick_pose,place_pose)
#            print ("Response: ",resp.success)
#        except rospy.ServiceException, e:
#            print "Service failed: %s"%e
    filename = 'output_'+str(test_scene_num.data)+'.yaml'
    send_to_yaml(filename, dict_list)
    return
    
if __name__ == '__main__':
    # Initialize color_list
    get_color_list.color_list = []

    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2,pcl_callback, queue_size=1)

    # Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pr2_base_mover_pub = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10) 

    # Load Model From disk
    model = pickle.load(open("model.sav", "rb"))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    #get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
     rospy.spin()

