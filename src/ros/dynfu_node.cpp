// ros & pcl
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/String.h>
// cugl
#include <cugl_ros/apps/dynamic_fusion/dynfu_app.h>

int main(int argc, char **argv)
{
  dynfu::DynfuApp dynfu_app(17, 520, 520);

  if (dynfu_app.init(argc, argv))
  {
    dynfu_app.startMainLoop();

    ros::init(argc, argv, "dynfu_node");

    ros::NodeHandle nh;
    ros::Publisher dynfu_pub = nh.advertise<sensor_msgs::PointCloud2>("dynfu_cloud", 1);
    ros::Publisher string_pub = nh.advertise<std_msgs::String>("dynfu_string", 5);

    ros::Rate loop_rate(5);

    int count = 0;

    while (ros::ok() && !dynfu_app.is_stopped())
    {
      // size_t surfel_size = dynfu_app.getSurfelSize();

      // if (surfel_size > 0)
      // {
      //   pcl::PointCloud<pcl::PointXYZRGB> cloud;

      //   cloud.width = surfel_size;
      //   cloud.height = 1;
      //   cloud.points.resize(cloud.width * cloud.height);

      //   if (dynfu_app.getSurfelDataHost(surfel_size, reinterpret_cast<float *>(cloud.points.data())))
      //   {
      //     sensor_msgs::PointCloud2 output;

      //     pcl::toROSMsg(cloud, output);
      //     output.header.frame_id = /*"camera_depth_optical_frame"*/"map";
      //     output.header.stamp = ros::Time::now();

      //     dynfu_pub.publish(output);

      //     ros::spinOnce();
      //   }
      // }

      std_msgs::String msg;

      std::stringstream ss;
      ss << "hello world";
      msg.data = ss.str();

      string_pub.publish(msg);

      ros::spinOnce();

      std::cout << "count = " << (++count) << std::endl;

      loop_rate.sleep();
    }

    return 0;
  }

  return -1;
}
