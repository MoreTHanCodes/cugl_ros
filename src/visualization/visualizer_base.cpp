#include <cugl_ros/visualization/visualizer_base.h>

namespace cugl
{
VisualizerBase *VisualizerBase::instance_ = NULL;

VisualizerBase::VisualizerBase(unsigned int refresh_delay)
    : refresh_delay_(refresh_delay)
{
}

VisualizerBase::~VisualizerBase()
{
}

bool VisualizerBase::init(int argc, char **argv)
{
  return true;
}

void VisualizerBase::startMainLoop()
{
}

void VisualizerBase::setInstance()
{
  instance_ = this;
}

void VisualizerBase::idle()
{
}

void VisualizerBase::display()
{
}

void VisualizerBase::reshape(int width, int height)
{
}

void VisualizerBase::keyboard(unsigned char key, int x, int y)
{
  switch(key)
  {
    case 27:
      glutDestroyWindow(glutGetWindow());
      break;
  }
}

void VisualizerBase::mouse(int button, int state, int x, int y)
{
}

void VisualizerBase::motion(int x, int y)
{
}

void VisualizerBase::timerEvent(int value)
{
  int cur_win = glutGetWindow();

  if(cur_win)
  {
    glutSetWindow(cur_win);
    glutPostRedisplay();
    glutTimerFunc(refresh_delay_, timerEventWrapper, 0);
  }
}

void VisualizerBase::idleWrapper()
{
  instance_->idle();
}

void VisualizerBase::displayWrapper()
{
  instance_->display();
}

void VisualizerBase::reshapeWrapper(int width, int height)
{
  instance_->reshape(width, height);
}

void VisualizerBase::keyboardWrapper(unsigned char key, int x, int y)
{
  instance_->keyboard(key, x, y);
}

void VisualizerBase::mouseWrapper(int button, int state, int x, int y)
{
  instance_->mouse(button, state, x, y);
}

void VisualizerBase::motionWrapper(int x, int y)
{
  instance_->motion(x, y);
}

void VisualizerBase::timerEventWrapper(int value)
{
  instance_->timerEvent(value);
}
} // namespace
