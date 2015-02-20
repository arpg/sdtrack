// Copyright (c) George Washington University, all rights reserved.  See the
// accompanying LICENSE file for more information.

// TODO let's move this to SceneGraph

#pragma once

#include <vector>

#include <Eigen/Core>

#include <SceneGraph/SceneGraph.h>
#include <pangolin/gldraw.h>


namespace Eigen {

#define USING_VECTOR_ARRAY(size)                                \
  using Vector##size##tArray =                                  \
      std::vector< Matrix<double,size,1>,                       \
                   Eigen::aligned_allocator<Matrix<double,size,1>>>;

USING_VECTOR_ARRAY(2);
USING_VECTOR_ARRAY(3);
USING_VECTOR_ARRAY(4);
USING_VECTOR_ARRAY(5);
USING_VECTOR_ARRAY(6);

#undef USING_VECTOR_ARRAY
}



template<typename Scalar>
inline void glMultMatrixT(const Scalar* data);
template<typename Scalar>
inline void glTranslateT(const Scalar x, const Scalar y,
                         const Scalar z);

#ifndef HAVE_GLES

template<>
inline void glMultMatrixT<double>(const double* data )
{
    glMultMatrixd(data);
}

template<>
inline void glTranslateT<double>(const double x, const double y,
                                 const double z)
{
  glTranslated(x,y,z);
}

#endif  // HAVE_GLES

template<>
inline void glMultMatrixT<float>(const float* data )
{
    glMultMatrixf(data);
}

template<>
inline void glTranslateT<float>(const float x, const float y,
                                  const float z)
{
    glTranslatef(x,y,z);
}

inline void draw2dLines( const Eigen::Vector2tArray& pts )
{
    // TODO fix this
//    glVertexPointer( 2, GL_FLOAT, 2, &pts[0][0] );
//    glEnableClientState( GL_VERTEX_ARRAY );
//    glDrawArrays( GL_LINES, 0, pts.size() );
//    glDisableClientState( GL_VERTEX_ARRAY );
    for( size_t ii = 1; ii < pts.size(); ii++ ){
        pangolin::glDrawLine( pts[ii-1][0], pts[ii-1][1],pts[ii][0], pts[ii][1] );
    }
}

////////////////////////////////////////////////////////////////////////////////
inline void glDrawEllipse(int segments, GLfloat width, GLfloat height,
                          GLfloat x1, GLfloat y1, bool filled)
{
    glPushMatrix();
    glTranslateT(x1, y1, 0.0f);
    GLfloat vertices[segments*2];
    GLfloat factor = M_PI / 180.0f;
    int count=0;
    for( GLfloat i = 0; i < 360.0f; i+=(360.0f/segments) ){
        vertices[count++] = (cos(factor*i)*width);
        vertices[count++] = (sin(factor*i)*height);
    }
    glVertexPointer( 2, GL_FLOAT , 0, vertices );
    glEnableClientState( GL_VERTEX_ARRAY );
    glDrawArrays( (filled) ? GL_TRIANGLE_FAN : GL_LINE_LOOP, 0, segments );
    glDisableClientState( GL_VERTEX_ARRAY );

    glPopMatrix();
}

////////////////////////////////////////////////////////////////////////////////
inline void
drawCircle(GLfloat x1, GLfloat y1, GLfloat radius_x, GLfloat radius_y,
           bool filled = false)
{
    glDrawEllipse(30, radius_x, radius_y, x1, y1, filled);
}

////////////////////////////////////////////////////////////////////////////////

inline void
drawSolidCircle(double x1, double y1, double radius_x, double radius_y)
{
    drawCircle(x1, y1, radius_x, radius_y,  true);
}

////////////////////////////////////////////////////////////////////////////////
// parallel to the x-z plane
inline void
drawCircle3D(double x1, double y1,  double radius, double z)
{
    int segments = 30;
    int count = 0;
    GLfloat vertices[segments*3];
    glTranslateT(x1, y1, z);
    for(int angle = 0; angle <= 360; angle+=(360 / segments))
    {
        double radians = M_PI*(double)angle/180;
        vertices[count++] = x1 + sin(radians) * radius;
        vertices[count++] = y1 + cos(radians) * radius;
        vertices[count++] = z;
    }
    glVertexPointer (3, GL_FLOAT , 0, vertices);
    glDrawArrays (GL_LINE_LOOP, 0, segments);
}

////////////////////////////////////////////////////////////////////////////////
// Parallel to the x,y,z axis
inline void
glDrawRectPerimeter3D(double x1, double y1, double x2, double y2, double z)
{
    pangolin::glDrawLine(x1,y1,z,x1,y2,z);
    pangolin::glDrawLine(x1,y1,z,x2,y1,z);
    pangolin::glDrawLine(x2,y2,z,x1,y2,z);
    pangolin::glDrawLine(x2,y2,z,x2,y1,z);
}

////////////////////////////////////////////////////////////////////////////////
// Parallel to the x,y,z axis
inline void
drawWiredBox(double x1, double y1, double z1, double x2, double y2, double z2)
{
    glDrawRectPerimeter3D(x1,y1,x2,y2,z1);
    glDrawRectPerimeter3D(x1,y1,x2,y2,z2);
    pangolin::glDrawLine(x1,y1,z1,x1,y1,z2);
    pangolin::glDrawLine(x2,y2,z1,x2,y2,z2);
    pangolin::glDrawLine(x1,y2,z1,x1,y2,z2);
    pangolin::glDrawLine(x2,y1,z1,x2,y1,z2);
}

////////////////////////////////////////////////////////////////////////////////
inline void drawPoint( float x, float y)
{
    GLfloat verts[] = { x,y };
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 2, GL_FLOAT, 0, verts );
    glDrawArrays( GL_POINTS, 0, 1 );
    glDisableClientState( GL_VERTEX_ARRAY );
}

////////////////////////////////////////////////////////////////////////////////
inline void drawPoint( float x, float y, float z)
{
    GLfloat verts[] = { x, y, z };
    glEnableClientState( GL_VERTEX_ARRAY );
    glVertexPointer( 3, GL_FLOAT, 0, verts );
    glDrawArrays( GL_POINTS, 0, 1 );
    glDisableClientState( GL_VERTEX_ARRAY );
}

////////////////////////////////////////////////////////////////////////////////
inline void
glDrawRectPerimeter( const double x, const double y, const double width,
                     const double height, const double dOrientation)
{
    glPushMatrix();
    glMultMatrixT(SceneGraph::GLCart2T(x,y,0.0,0.0,0.0,dOrientation).data());
    pangolin::glDrawLine(0,0,width/2,0);
    pangolin::glDrawLine(-width/2,-height/2,-width/2,height/2);
    pangolin::glDrawLine(-width/2,-height/2,width/2,-height/2);
    pangolin::glDrawLine(width/2,height/2,-width/2,height/2);
    pangolin::glDrawLine(width/2,height/2,width/2,-height/2);
    glPopMatrix();
}

////////////////////////////////////////////////////////////////////////////////
inline void glDrawPolygon2d( const GLfloat* verts, const size_t nverts)
{
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_FLOAT, 0, verts);
    glDrawArrays(GL_TRIANGLE_FAN, 0, nverts);
    glDisableClientState(GL_VERTEX_ARRAY);
}
