// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

#pragma once

#include <vector>
#include <deque>
#include <string>
#include <mutex>
#include <iomanip>

#include <pangolin/pangolin.h>
#include <SceneGraph/SceneGraph.h>

#include "color_palette.h"
#include "gl_simple_object.h"

class TimerView : public pangolin::View
{
 public:

  TimerView():
      is_timer_init_(false),
      offset_y_(0),
      time_window_size_(10),
      fps_str_(""),
      fps_(0.0),
      history_max_(0.0),
      percent_graphics_(0.5),
      percent_function_name_(0.8),
      division_(0),
      step_y_(0),
      step_x_(0),
      time_offset_(0),
      width_(0),
      height_(0),
      border_offset_(2.0) {}

  ~TimerView(){}

  void InitReset() {
    is_timer_init_  = false;
    offset_y_    = 13;
    names_.clear();
    times_.clear();
    history_.clear();
  }

  void Update(int nTimeWindowSize,
              std::vector< std::string >&  vsNames,
              std::vector< std::pair<double,double> >& vTimes) {

    std::unique_lock<std::mutex> lock(mutex_);

    if(vTimes.empty()){
      return;
    }

    step_y_ = height_/vsNames.size();
    step_x_ = division_/nTimeWindowSize;

    // fill our member variables with updated values
    time_window_size_ = nTimeWindowSize;
    names_         = vsNames;

    std::vector<double> vdAccTimes;
    times_.resize(vsNames.size());

    // init accTimes and string vector
    for(int ii=0; ii <(int)vTimes.size(); ++ii) {
      vdAccTimes.push_back(vTimes[ii].second);
      std::stringstream ss;
      ss << std::setprecision(2) << vTimes[ii].first;
      times_[ii] = ss.str();
    }

    //compute accumulated processing time
    for(int ii=(int)vdAccTimes.size()-2; ii >= 0; --ii){
      vdAccTimes[ii] += vdAccTimes[ii+1];
    }

    // save new times in the history
    history_.push_back(vdAccTimes);

    // check if history queue is full
    if((int)history_.size() > nTimeWindowSize){
      history_.pop_front();
    }

    history_max_ = 0.0;

    if(!history_.empty()){

      fps_ = floor((0.8)*fps_ + (0.2)*(1000.0 / history_.back().front()));

      std::stringstream ss;
      ss.setf( std::ios::fixed, std:: ios::floatfield );
      ss << "FPS: " << std::setprecision(1) << fps_;
      fps_str_ = ss.str();

      // compute max value for scaling
      for(int ii = 0; ii < (int)history_.size(); ii++){
        if(history_[ii][0] > history_max_){
          history_max_ = history_[ii][0];
        }
      }
    }
  }

  // overloaded from View
  virtual void Resize(const pangolin::Viewport& parent) {

    pangolin::View::Resize(parent);

    ortho_ = pangolin::ProjectionMatrixOrthographic(0, v.w, v.h, 0, 0, 1E4);

    // recompute these for adjusting rendering
    width_      = (float)v.w - border_offset_*2;
    height_     = (float)v.h - border_offset_*2;
    division_   = width_ * percent_graphics_;
    time_offset_ = division_ +
        (width_ - division_) * percent_function_name_;
    step_y_      = ((int)names_.size() > 0)?height_/names_.size():0.0;
    step_x_      = division_/time_window_size_;
  }

  // overloaded from View
  virtual void Render() {
    std::unique_lock<std::mutex> lock(mutex_);

    pangolin::GlState state;

    state.glDisable(GL_DEPTH_TEST);
    state.glEnable(GL_BLEND);
    //state.glDisable(GL_LIGHTING);

    // Activate viewport
    this->Activate();

    // Load orthographic projection matrix to match image
    glMatrixMode(GL_PROJECTION);
    ortho_.Load();

    // Reset ModelView matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // draw widget bounding box BG
    glColor4f(0.2f,0.2f,0.2f,0.5f);
    GLfloat vertices[8];
    GLfloat x1,x2,y1,y2;
    x1 = border_offset_;
    y1 = border_offset_;
    x2 = border_offset_ + width_;
    y2 = border_offset_ + height_;
    vertices[0] = x2;
    vertices[1] = y2;
    vertices[2] = x2;
    vertices[3] = y1;
    vertices[4] = x1;
    vertices[5] = y1;
    vertices[6] = x1;
    vertices[7] = y2;

    glDrawPolygon2d(vertices, 4);

    // draw timed functions names and process time
    Palette& colors = color_palette_.GetPaletteRef(eOblivion);

    for(size_t ii=0; ii<names_.size(); ++ii) {
      size_t cidx = ii % (colors.size()-1);
      glColor4f(colors[cidx][0], colors[cidx][1], colors[cidx][2], 1.0);
      const GLfloat x = division_ + border_offset_ + 5.0;
      const GLfloat y = offset_y_ + step_y_*ii;
      gl_text_.Draw(names_[ii], x, y);
      glColor4f(1.0, 1.0, 1.0, 1.0);
      gl_text_.Draw(times_[ii],time_offset_ + border_offset_, y);
    }

    // draw process time history
    float fVX1, fVX2, fVX3, fVX4;
    float fVY1, fVY2, fVY3 = 0, fVY4 = 0;

    float fScale = 0.0;

    if(history_max_ > 0.0)
      fScale = height_/history_max_;

    size_t nInit = time_window_size_ - history_.size();

    for(size_t ii=0; ii<names_.size(); ++ii) {
      size_t cidx = ii % (colors.size()-1);
      glColor4f(colors[cidx][0], colors[cidx][1], colors[cidx][2], 0.8);
      for(size_t jj=nInit; jj<time_window_size_-1; ++jj) {
        fVX1 = border_offset_ + jj*step_x_;        // bottom-left
        fVY1 = height_ + border_offset_;
        fVX2 = border_offset_ + jj*step_x_;        // top-left
        fVY2 = height_ + border_offset_ - history_[jj-nInit][ii]*fScale;
        fVX3 = border_offset_ + (jj+1)*step_x_;    // top-right
        fVY3 = height_ + border_offset_ - history_[jj-nInit+1][ii]*fScale;
        fVX4 = border_offset_ + (jj+1)*step_x_;    // bottom-right
        fVY4 = height_ + border_offset_;

        if(ii < names_.size()-1) {
          fVY1 -= history_[jj-nInit][ii+1]*fScale;
          fVY4 -= history_[jj-nInit+1][ii+1]*fScale;
        }

        vertices[0] = fVX1;
        vertices[1] = fVY1;
        vertices[2] = fVX4;
        vertices[3] = fVY4;
        vertices[4] = fVX3;
        vertices[5] = fVY3;
        vertices[6] = fVX2;
        vertices[7] = fVY2;

        glDrawPolygon2d(vertices, 4);
      }

      fVX1 = border_offset_ + step_x_*(time_window_size_-1);
      fVX2 = fVX1;
      fVX3 = border_offset_ + step_x_*time_window_size_;
      fVY1 = fVY4;
      fVY2 = fVY3;
      fVY3 = offset_y_ + ii*step_y_ - step_y_*0.3;

      vertices[0] = fVX1;
      vertices[1] = fVY1;
      vertices[2] = fVX3;
      vertices[3] = fVY3;
      vertices[4] = fVX2;
      vertices[5] = fVY2;

      glDrawPolygon2d(vertices, 3);
    }

    // draw widget bounding box
    glLineWidth(1.0f);
    glColor4f(1.0,1.0,1.0,1.0);
    pangolin::glDrawRectPerimeter(
        border_offset_, border_offset_,
        border_offset_+width_, border_offset_+height_);

    // draw division between function names and graphics
    pangolin::glDrawLine(division_, border_offset_, division_, height_);

    // draw frames per second
    gl_text_.Draw(fps_str_, 5, offset_y_);

    // Call base View implementation
    pangolin::View::Render();
  }

 private:

  // Projection matrix
  pangolin::OpenGlMatrix ortho_;

  bool             is_timer_init_;
  int              offset_y_;
  size_t           time_window_size_;
  std::string      fps_str_;
  float            fps_;
  float            history_max_;

  // Render variables
  float               percent_graphics_;
  float               percent_function_name_;
  float               division_;
  float               step_y_;
  float               step_x_;
  float               time_offset_;
  float               width_;
  float               height_;
  float               border_offset_;
  // Need to implement a test to determine if using GLES

  SceneGraph::GLText  gl_text_;
  ColorPalette        color_palette_;

  std::vector< std::string > names_;
  std::vector< std::string > times_;
  std::deque< std::vector<double>  > history_;


  std::mutex  mutex_;
};
