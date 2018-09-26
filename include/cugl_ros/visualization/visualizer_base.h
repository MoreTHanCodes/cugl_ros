#ifndef __CUGL_VISUALIZATION_VISUALIZER_BASE_H__
#define __CUGL_VISUALIZATION_VISUALIZER_BASE_H__

#include <GL/freeglut.h>

#include <cugl_ros/common/containers/texture_bridge.h>

#include <string>
#include <iostream>

namespace cugl
{
/*
 * Base class that can be used to create a graphics application
 * that requires an animation loop.
 */

class VisualizerBase
{
 public:
  VisualizerBase(unsigned int refresh_delay = 33);
  virtual ~VisualizerBase();

  virtual bool init(int argc, char **argv);

  virtual void startMainLoop();

 protected:
  unsigned int refresh_delay_; // in milliseconds

  // Sets the instance to itself, used for callback wrapper functions registration
  static VisualizerBase *instance_;
  void setInstance();

  // rendering callbacks
  virtual void idle();
  virtual void display();
  virtual void reshape(int width, int height);
  virtual void keyboard(unsigned char key, int x, int y);
  virtual void mouse(int button, int state, int x, int y);
  virtual void motion(int x, int y);
  virtual void timerEvent(int value);

  static void idleWrapper();
  static void displayWrapper();
  static void reshapeWrapper(int width, int height);
  static void keyboardWrapper(unsigned char key, int x, int y);
  static void mouseWrapper(int button, int state, int x, int y);
  static void motionWrapper(int x, int y);
  static void timerEventWrapper(int value);
};
} // namespace

#endif /* __CUGL_VISUALIZATION_VISUALIZER_BASE_H__ */
