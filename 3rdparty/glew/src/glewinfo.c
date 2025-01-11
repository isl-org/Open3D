/*
** The OpenGL Extension Wrangler Library
** Copyright (C) 2008-2019, Nigel Stewart <nigels[]users sourceforge net>
** Copyright (C) 2002-2008, Milan Ikits <milan ikits[]ieee org>
** Copyright (C) 2002-2008, Marcelo E. Magallon <mmagallo[]debian org>
** Copyright (C) 2002, Lev Povalahev
** All rights reserved.
** 
** Redistribution and use in source and binary forms, with or without 
** modification, are permitted provided that the following conditions are met:
** 
** * Redistributions of source code must retain the above copyright notice, 
**   this list of conditions and the following disclaimer.
** * Redistributions in binary form must reproduce the above copyright notice, 
**   this list of conditions and the following disclaimer in the documentation 
**   and/or other materials provided with the distribution.
** * The name of the author may be used to endorse or promote products 
**   derived from this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
** ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
** LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
** CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
** SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
** INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
** CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
** ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
** THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#if defined(GLEW_EGL)
#include <GL/eglew.h>
#elif defined(GLEW_OSMESA)
#define GLAPI extern
#include <GL/osmesa.h>
#elif defined(_WIN32)
#include <GL/wglew.h>
#elif !defined(__APPLE__) && !defined(__HAIKU__) || defined(GLEW_APPLE_GLX)
#include <GL/glxew.h>
#endif

#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif

#ifdef GLEW_REGAL
#include <GL/Regal.h>
#endif

static FILE* f;

/* Command-line parameters for GL context creation */

struct createParams
{
#if defined(GLEW_OSMESA)
#elif defined(GLEW_EGL)
#elif defined(_WIN32)
  int         pixelformat;
#elif !defined(__APPLE__) && !defined(__HAIKU__) || defined(GLEW_APPLE_GLX)
  const char* display;
  int         visual;
#endif
  int         major, minor;  /* GL context version number */

  /* https://www.opengl.org/registry/specs/ARB/glx_create_context.txt */
  int         profile;       /* core = 1, compatibility = 2 */
  int         flags;         /* debug = 1, forward compatible = 2 */

  /* GLEW experimental mode */
  int         experimental;
};

GLboolean glewCreateContext (struct createParams *params);

GLboolean glewParseArgs (int argc, char** argv, struct createParams *);

void glewDestroyContext ();

/* ------------------------------------------------------------------------- */

static GLboolean glewPrintExt (const char* name, GLboolean def1, GLboolean def2, GLboolean def3)
{
  unsigned int i;
  fprintf(f, "\n%s:", name);
  for (i=0; i<62-strlen(name); i++) fprintf(f, " ");
  fprintf(f, "%s ", def1 ? "OK" : "MISSING");
  if (def1 != def2)
    fprintf(f, "[%s] ", def2 ? "OK" : "MISSING");
  if (def1 != def3)
    fprintf(f, "[%s]\n", def3 ? "OK" : "MISSING");
  else
    fprintf(f, "\n");
  for (i=0; i<strlen(name)+1; i++) fprintf(f, "-");
  fprintf(f, "\n");
  fflush(f);
  return def1 || def2 || def3 || glewExperimental; /* Enable per-function info too? */
}

static void glewInfoFunc (GLboolean fi, const char* name, GLint undefined)
{
  unsigned int i;
  if (fi)
  {
    fprintf(f, "  %s:", name);
    for (i=0; i<60-strlen(name); i++) fprintf(f, " ");
    fprintf(f, "%s\n", undefined ? "MISSING" : "OK");
    fflush(f);
  }
}

/* ----------------------------- GL_VERSION_1_1 ---------------------------- */

#ifdef GL_VERSION_1_1

static void _glewInfo_GL_VERSION_1_1 (void)
{
  glewPrintExt("GL_VERSION_1_1", GLEW_VERSION_1_1, GLEW_VERSION_1_1, GLEW_VERSION_1_1);
}

#endif /* GL_VERSION_1_1 */

#ifdef GL_VERSION_1_2

static void _glewInfo_GL_VERSION_1_2 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_1_2", GLEW_VERSION_1_2, GLEW_VERSION_1_2, GLEW_VERSION_1_2);

  glewInfoFunc(fi, "glCopyTexSubImage3D", glCopyTexSubImage3D == NULL);
  glewInfoFunc(fi, "glDrawRangeElements", glDrawRangeElements == NULL);
  glewInfoFunc(fi, "glTexImage3D", glTexImage3D == NULL);
  glewInfoFunc(fi, "glTexSubImage3D", glTexSubImage3D == NULL);
}

#endif /* GL_VERSION_1_2 */

#ifdef GL_VERSION_1_2_1

static void _glewInfo_GL_VERSION_1_2_1 (void)
{
  glewPrintExt("GL_VERSION_1_2_1", GLEW_VERSION_1_2_1, GLEW_VERSION_1_2_1, GLEW_VERSION_1_2_1);
}

#endif /* GL_VERSION_1_2_1 */

#ifdef GL_VERSION_1_3

static void _glewInfo_GL_VERSION_1_3 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_1_3", GLEW_VERSION_1_3, GLEW_VERSION_1_3, GLEW_VERSION_1_3);

  glewInfoFunc(fi, "glActiveTexture", glActiveTexture == NULL);
  glewInfoFunc(fi, "glClientActiveTexture", glClientActiveTexture == NULL);
  glewInfoFunc(fi, "glCompressedTexImage1D", glCompressedTexImage1D == NULL);
  glewInfoFunc(fi, "glCompressedTexImage2D", glCompressedTexImage2D == NULL);
  glewInfoFunc(fi, "glCompressedTexImage3D", glCompressedTexImage3D == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage1D", glCompressedTexSubImage1D == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage2D", glCompressedTexSubImage2D == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage3D", glCompressedTexSubImage3D == NULL);
  glewInfoFunc(fi, "glGetCompressedTexImage", glGetCompressedTexImage == NULL);
  glewInfoFunc(fi, "glLoadTransposeMatrixd", glLoadTransposeMatrixd == NULL);
  glewInfoFunc(fi, "glLoadTransposeMatrixf", glLoadTransposeMatrixf == NULL);
  glewInfoFunc(fi, "glMultTransposeMatrixd", glMultTransposeMatrixd == NULL);
  glewInfoFunc(fi, "glMultTransposeMatrixf", glMultTransposeMatrixf == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1d", glMultiTexCoord1d == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1dv", glMultiTexCoord1dv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1f", glMultiTexCoord1f == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1fv", glMultiTexCoord1fv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1i", glMultiTexCoord1i == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1iv", glMultiTexCoord1iv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1s", glMultiTexCoord1s == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1sv", glMultiTexCoord1sv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2d", glMultiTexCoord2d == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2dv", glMultiTexCoord2dv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2f", glMultiTexCoord2f == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2fv", glMultiTexCoord2fv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2i", glMultiTexCoord2i == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2iv", glMultiTexCoord2iv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2s", glMultiTexCoord2s == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2sv", glMultiTexCoord2sv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3d", glMultiTexCoord3d == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3dv", glMultiTexCoord3dv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3f", glMultiTexCoord3f == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3fv", glMultiTexCoord3fv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3i", glMultiTexCoord3i == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3iv", glMultiTexCoord3iv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3s", glMultiTexCoord3s == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3sv", glMultiTexCoord3sv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4d", glMultiTexCoord4d == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4dv", glMultiTexCoord4dv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4f", glMultiTexCoord4f == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4fv", glMultiTexCoord4fv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4i", glMultiTexCoord4i == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4iv", glMultiTexCoord4iv == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4s", glMultiTexCoord4s == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4sv", glMultiTexCoord4sv == NULL);
  glewInfoFunc(fi, "glSampleCoverage", glSampleCoverage == NULL);
}

#endif /* GL_VERSION_1_3 */

#ifdef GL_VERSION_1_4

static void _glewInfo_GL_VERSION_1_4 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_1_4", GLEW_VERSION_1_4, GLEW_VERSION_1_4, GLEW_VERSION_1_4);

  glewInfoFunc(fi, "glBlendColor", glBlendColor == NULL);
  glewInfoFunc(fi, "glBlendEquation", glBlendEquation == NULL);
  glewInfoFunc(fi, "glBlendFuncSeparate", glBlendFuncSeparate == NULL);
  glewInfoFunc(fi, "glFogCoordPointer", glFogCoordPointer == NULL);
  glewInfoFunc(fi, "glFogCoordd", glFogCoordd == NULL);
  glewInfoFunc(fi, "glFogCoorddv", glFogCoorddv == NULL);
  glewInfoFunc(fi, "glFogCoordf", glFogCoordf == NULL);
  glewInfoFunc(fi, "glFogCoordfv", glFogCoordfv == NULL);
  glewInfoFunc(fi, "glMultiDrawArrays", glMultiDrawArrays == NULL);
  glewInfoFunc(fi, "glMultiDrawElements", glMultiDrawElements == NULL);
  glewInfoFunc(fi, "glPointParameterf", glPointParameterf == NULL);
  glewInfoFunc(fi, "glPointParameterfv", glPointParameterfv == NULL);
  glewInfoFunc(fi, "glPointParameteri", glPointParameteri == NULL);
  glewInfoFunc(fi, "glPointParameteriv", glPointParameteriv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3b", glSecondaryColor3b == NULL);
  glewInfoFunc(fi, "glSecondaryColor3bv", glSecondaryColor3bv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3d", glSecondaryColor3d == NULL);
  glewInfoFunc(fi, "glSecondaryColor3dv", glSecondaryColor3dv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3f", glSecondaryColor3f == NULL);
  glewInfoFunc(fi, "glSecondaryColor3fv", glSecondaryColor3fv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3i", glSecondaryColor3i == NULL);
  glewInfoFunc(fi, "glSecondaryColor3iv", glSecondaryColor3iv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3s", glSecondaryColor3s == NULL);
  glewInfoFunc(fi, "glSecondaryColor3sv", glSecondaryColor3sv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3ub", glSecondaryColor3ub == NULL);
  glewInfoFunc(fi, "glSecondaryColor3ubv", glSecondaryColor3ubv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3ui", glSecondaryColor3ui == NULL);
  glewInfoFunc(fi, "glSecondaryColor3uiv", glSecondaryColor3uiv == NULL);
  glewInfoFunc(fi, "glSecondaryColor3us", glSecondaryColor3us == NULL);
  glewInfoFunc(fi, "glSecondaryColor3usv", glSecondaryColor3usv == NULL);
  glewInfoFunc(fi, "glSecondaryColorPointer", glSecondaryColorPointer == NULL);
  glewInfoFunc(fi, "glWindowPos2d", glWindowPos2d == NULL);
  glewInfoFunc(fi, "glWindowPos2dv", glWindowPos2dv == NULL);
  glewInfoFunc(fi, "glWindowPos2f", glWindowPos2f == NULL);
  glewInfoFunc(fi, "glWindowPos2fv", glWindowPos2fv == NULL);
  glewInfoFunc(fi, "glWindowPos2i", glWindowPos2i == NULL);
  glewInfoFunc(fi, "glWindowPos2iv", glWindowPos2iv == NULL);
  glewInfoFunc(fi, "glWindowPos2s", glWindowPos2s == NULL);
  glewInfoFunc(fi, "glWindowPos2sv", glWindowPos2sv == NULL);
  glewInfoFunc(fi, "glWindowPos3d", glWindowPos3d == NULL);
  glewInfoFunc(fi, "glWindowPos3dv", glWindowPos3dv == NULL);
  glewInfoFunc(fi, "glWindowPos3f", glWindowPos3f == NULL);
  glewInfoFunc(fi, "glWindowPos3fv", glWindowPos3fv == NULL);
  glewInfoFunc(fi, "glWindowPos3i", glWindowPos3i == NULL);
  glewInfoFunc(fi, "glWindowPos3iv", glWindowPos3iv == NULL);
  glewInfoFunc(fi, "glWindowPos3s", glWindowPos3s == NULL);
  glewInfoFunc(fi, "glWindowPos3sv", glWindowPos3sv == NULL);
}

#endif /* GL_VERSION_1_4 */

#ifdef GL_VERSION_1_5

static void _glewInfo_GL_VERSION_1_5 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_1_5", GLEW_VERSION_1_5, GLEW_VERSION_1_5, GLEW_VERSION_1_5);

  glewInfoFunc(fi, "glBeginQuery", glBeginQuery == NULL);
  glewInfoFunc(fi, "glBindBuffer", glBindBuffer == NULL);
  glewInfoFunc(fi, "glBufferData", glBufferData == NULL);
  glewInfoFunc(fi, "glBufferSubData", glBufferSubData == NULL);
  glewInfoFunc(fi, "glDeleteBuffers", glDeleteBuffers == NULL);
  glewInfoFunc(fi, "glDeleteQueries", glDeleteQueries == NULL);
  glewInfoFunc(fi, "glEndQuery", glEndQuery == NULL);
  glewInfoFunc(fi, "glGenBuffers", glGenBuffers == NULL);
  glewInfoFunc(fi, "glGenQueries", glGenQueries == NULL);
  glewInfoFunc(fi, "glGetBufferParameteriv", glGetBufferParameteriv == NULL);
  glewInfoFunc(fi, "glGetBufferPointerv", glGetBufferPointerv == NULL);
  glewInfoFunc(fi, "glGetBufferSubData", glGetBufferSubData == NULL);
  glewInfoFunc(fi, "glGetQueryObjectiv", glGetQueryObjectiv == NULL);
  glewInfoFunc(fi, "glGetQueryObjectuiv", glGetQueryObjectuiv == NULL);
  glewInfoFunc(fi, "glGetQueryiv", glGetQueryiv == NULL);
  glewInfoFunc(fi, "glIsBuffer", glIsBuffer == NULL);
  glewInfoFunc(fi, "glIsQuery", glIsQuery == NULL);
  glewInfoFunc(fi, "glMapBuffer", glMapBuffer == NULL);
  glewInfoFunc(fi, "glUnmapBuffer", glUnmapBuffer == NULL);
}

#endif /* GL_VERSION_1_5 */

#ifdef GL_VERSION_2_0

static void _glewInfo_GL_VERSION_2_0 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_2_0", GLEW_VERSION_2_0, GLEW_VERSION_2_0, GLEW_VERSION_2_0);

  glewInfoFunc(fi, "glAttachShader", glAttachShader == NULL);
  glewInfoFunc(fi, "glBindAttribLocation", glBindAttribLocation == NULL);
  glewInfoFunc(fi, "glBlendEquationSeparate", glBlendEquationSeparate == NULL);
  glewInfoFunc(fi, "glCompileShader", glCompileShader == NULL);
  glewInfoFunc(fi, "glCreateProgram", glCreateProgram == NULL);
  glewInfoFunc(fi, "glCreateShader", glCreateShader == NULL);
  glewInfoFunc(fi, "glDeleteProgram", glDeleteProgram == NULL);
  glewInfoFunc(fi, "glDeleteShader", glDeleteShader == NULL);
  glewInfoFunc(fi, "glDetachShader", glDetachShader == NULL);
  glewInfoFunc(fi, "glDisableVertexAttribArray", glDisableVertexAttribArray == NULL);
  glewInfoFunc(fi, "glDrawBuffers", glDrawBuffers == NULL);
  glewInfoFunc(fi, "glEnableVertexAttribArray", glEnableVertexAttribArray == NULL);
  glewInfoFunc(fi, "glGetActiveAttrib", glGetActiveAttrib == NULL);
  glewInfoFunc(fi, "glGetActiveUniform", glGetActiveUniform == NULL);
  glewInfoFunc(fi, "glGetAttachedShaders", glGetAttachedShaders == NULL);
  glewInfoFunc(fi, "glGetAttribLocation", glGetAttribLocation == NULL);
  glewInfoFunc(fi, "glGetProgramInfoLog", glGetProgramInfoLog == NULL);
  glewInfoFunc(fi, "glGetProgramiv", glGetProgramiv == NULL);
  glewInfoFunc(fi, "glGetShaderInfoLog", glGetShaderInfoLog == NULL);
  glewInfoFunc(fi, "glGetShaderSource", glGetShaderSource == NULL);
  glewInfoFunc(fi, "glGetShaderiv", glGetShaderiv == NULL);
  glewInfoFunc(fi, "glGetUniformLocation", glGetUniformLocation == NULL);
  glewInfoFunc(fi, "glGetUniformfv", glGetUniformfv == NULL);
  glewInfoFunc(fi, "glGetUniformiv", glGetUniformiv == NULL);
  glewInfoFunc(fi, "glGetVertexAttribPointerv", glGetVertexAttribPointerv == NULL);
  glewInfoFunc(fi, "glGetVertexAttribdv", glGetVertexAttribdv == NULL);
  glewInfoFunc(fi, "glGetVertexAttribfv", glGetVertexAttribfv == NULL);
  glewInfoFunc(fi, "glGetVertexAttribiv", glGetVertexAttribiv == NULL);
  glewInfoFunc(fi, "glIsProgram", glIsProgram == NULL);
  glewInfoFunc(fi, "glIsShader", glIsShader == NULL);
  glewInfoFunc(fi, "glLinkProgram", glLinkProgram == NULL);
  glewInfoFunc(fi, "glShaderSource", glShaderSource == NULL);
  glewInfoFunc(fi, "glStencilFuncSeparate", glStencilFuncSeparate == NULL);
  glewInfoFunc(fi, "glStencilMaskSeparate", glStencilMaskSeparate == NULL);
  glewInfoFunc(fi, "glStencilOpSeparate", glStencilOpSeparate == NULL);
  glewInfoFunc(fi, "glUniform1f", glUniform1f == NULL);
  glewInfoFunc(fi, "glUniform1fv", glUniform1fv == NULL);
  glewInfoFunc(fi, "glUniform1i", glUniform1i == NULL);
  glewInfoFunc(fi, "glUniform1iv", glUniform1iv == NULL);
  glewInfoFunc(fi, "glUniform2f", glUniform2f == NULL);
  glewInfoFunc(fi, "glUniform2fv", glUniform2fv == NULL);
  glewInfoFunc(fi, "glUniform2i", glUniform2i == NULL);
  glewInfoFunc(fi, "glUniform2iv", glUniform2iv == NULL);
  glewInfoFunc(fi, "glUniform3f", glUniform3f == NULL);
  glewInfoFunc(fi, "glUniform3fv", glUniform3fv == NULL);
  glewInfoFunc(fi, "glUniform3i", glUniform3i == NULL);
  glewInfoFunc(fi, "glUniform3iv", glUniform3iv == NULL);
  glewInfoFunc(fi, "glUniform4f", glUniform4f == NULL);
  glewInfoFunc(fi, "glUniform4fv", glUniform4fv == NULL);
  glewInfoFunc(fi, "glUniform4i", glUniform4i == NULL);
  glewInfoFunc(fi, "glUniform4iv", glUniform4iv == NULL);
  glewInfoFunc(fi, "glUniformMatrix2fv", glUniformMatrix2fv == NULL);
  glewInfoFunc(fi, "glUniformMatrix3fv", glUniformMatrix3fv == NULL);
  glewInfoFunc(fi, "glUniformMatrix4fv", glUniformMatrix4fv == NULL);
  glewInfoFunc(fi, "glUseProgram", glUseProgram == NULL);
  glewInfoFunc(fi, "glValidateProgram", glValidateProgram == NULL);
  glewInfoFunc(fi, "glVertexAttrib1d", glVertexAttrib1d == NULL);
  glewInfoFunc(fi, "glVertexAttrib1dv", glVertexAttrib1dv == NULL);
  glewInfoFunc(fi, "glVertexAttrib1f", glVertexAttrib1f == NULL);
  glewInfoFunc(fi, "glVertexAttrib1fv", glVertexAttrib1fv == NULL);
  glewInfoFunc(fi, "glVertexAttrib1s", glVertexAttrib1s == NULL);
  glewInfoFunc(fi, "glVertexAttrib1sv", glVertexAttrib1sv == NULL);
  glewInfoFunc(fi, "glVertexAttrib2d", glVertexAttrib2d == NULL);
  glewInfoFunc(fi, "glVertexAttrib2dv", glVertexAttrib2dv == NULL);
  glewInfoFunc(fi, "glVertexAttrib2f", glVertexAttrib2f == NULL);
  glewInfoFunc(fi, "glVertexAttrib2fv", glVertexAttrib2fv == NULL);
  glewInfoFunc(fi, "glVertexAttrib2s", glVertexAttrib2s == NULL);
  glewInfoFunc(fi, "glVertexAttrib2sv", glVertexAttrib2sv == NULL);
  glewInfoFunc(fi, "glVertexAttrib3d", glVertexAttrib3d == NULL);
  glewInfoFunc(fi, "glVertexAttrib3dv", glVertexAttrib3dv == NULL);
  glewInfoFunc(fi, "glVertexAttrib3f", glVertexAttrib3f == NULL);
  glewInfoFunc(fi, "glVertexAttrib3fv", glVertexAttrib3fv == NULL);
  glewInfoFunc(fi, "glVertexAttrib3s", glVertexAttrib3s == NULL);
  glewInfoFunc(fi, "glVertexAttrib3sv", glVertexAttrib3sv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4Nbv", glVertexAttrib4Nbv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4Niv", glVertexAttrib4Niv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4Nsv", glVertexAttrib4Nsv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4Nub", glVertexAttrib4Nub == NULL);
  glewInfoFunc(fi, "glVertexAttrib4Nubv", glVertexAttrib4Nubv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4Nuiv", glVertexAttrib4Nuiv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4Nusv", glVertexAttrib4Nusv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4bv", glVertexAttrib4bv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4d", glVertexAttrib4d == NULL);
  glewInfoFunc(fi, "glVertexAttrib4dv", glVertexAttrib4dv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4f", glVertexAttrib4f == NULL);
  glewInfoFunc(fi, "glVertexAttrib4fv", glVertexAttrib4fv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4iv", glVertexAttrib4iv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4s", glVertexAttrib4s == NULL);
  glewInfoFunc(fi, "glVertexAttrib4sv", glVertexAttrib4sv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4ubv", glVertexAttrib4ubv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4uiv", glVertexAttrib4uiv == NULL);
  glewInfoFunc(fi, "glVertexAttrib4usv", glVertexAttrib4usv == NULL);
  glewInfoFunc(fi, "glVertexAttribPointer", glVertexAttribPointer == NULL);
}

#endif /* GL_VERSION_2_0 */

#ifdef GL_VERSION_2_1

static void _glewInfo_GL_VERSION_2_1 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_2_1", GLEW_VERSION_2_1, GLEW_VERSION_2_1, GLEW_VERSION_2_1);

  glewInfoFunc(fi, "glUniformMatrix2x3fv", glUniformMatrix2x3fv == NULL);
  glewInfoFunc(fi, "glUniformMatrix2x4fv", glUniformMatrix2x4fv == NULL);
  glewInfoFunc(fi, "glUniformMatrix3x2fv", glUniformMatrix3x2fv == NULL);
  glewInfoFunc(fi, "glUniformMatrix3x4fv", glUniformMatrix3x4fv == NULL);
  glewInfoFunc(fi, "glUniformMatrix4x2fv", glUniformMatrix4x2fv == NULL);
  glewInfoFunc(fi, "glUniformMatrix4x3fv", glUniformMatrix4x3fv == NULL);
}

#endif /* GL_VERSION_2_1 */

#ifdef GL_VERSION_3_0

static void _glewInfo_GL_VERSION_3_0 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_3_0", GLEW_VERSION_3_0, GLEW_VERSION_3_0, GLEW_VERSION_3_0);

  glewInfoFunc(fi, "glBeginConditionalRender", glBeginConditionalRender == NULL);
  glewInfoFunc(fi, "glBeginTransformFeedback", glBeginTransformFeedback == NULL);
  glewInfoFunc(fi, "glBindFragDataLocation", glBindFragDataLocation == NULL);
  glewInfoFunc(fi, "glClampColor", glClampColor == NULL);
  glewInfoFunc(fi, "glClearBufferfi", glClearBufferfi == NULL);
  glewInfoFunc(fi, "glClearBufferfv", glClearBufferfv == NULL);
  glewInfoFunc(fi, "glClearBufferiv", glClearBufferiv == NULL);
  glewInfoFunc(fi, "glClearBufferuiv", glClearBufferuiv == NULL);
  glewInfoFunc(fi, "glColorMaski", glColorMaski == NULL);
  glewInfoFunc(fi, "glDisablei", glDisablei == NULL);
  glewInfoFunc(fi, "glEnablei", glEnablei == NULL);
  glewInfoFunc(fi, "glEndConditionalRender", glEndConditionalRender == NULL);
  glewInfoFunc(fi, "glEndTransformFeedback", glEndTransformFeedback == NULL);
  glewInfoFunc(fi, "glGetBooleani_v", glGetBooleani_v == NULL);
  glewInfoFunc(fi, "glGetFragDataLocation", glGetFragDataLocation == NULL);
  glewInfoFunc(fi, "glGetStringi", glGetStringi == NULL);
  glewInfoFunc(fi, "glGetTexParameterIiv", glGetTexParameterIiv == NULL);
  glewInfoFunc(fi, "glGetTexParameterIuiv", glGetTexParameterIuiv == NULL);
  glewInfoFunc(fi, "glGetTransformFeedbackVarying", glGetTransformFeedbackVarying == NULL);
  glewInfoFunc(fi, "glGetUniformuiv", glGetUniformuiv == NULL);
  glewInfoFunc(fi, "glGetVertexAttribIiv", glGetVertexAttribIiv == NULL);
  glewInfoFunc(fi, "glGetVertexAttribIuiv", glGetVertexAttribIuiv == NULL);
  glewInfoFunc(fi, "glIsEnabledi", glIsEnabledi == NULL);
  glewInfoFunc(fi, "glTexParameterIiv", glTexParameterIiv == NULL);
  glewInfoFunc(fi, "glTexParameterIuiv", glTexParameterIuiv == NULL);
  glewInfoFunc(fi, "glTransformFeedbackVaryings", glTransformFeedbackVaryings == NULL);
  glewInfoFunc(fi, "glUniform1ui", glUniform1ui == NULL);
  glewInfoFunc(fi, "glUniform1uiv", glUniform1uiv == NULL);
  glewInfoFunc(fi, "glUniform2ui", glUniform2ui == NULL);
  glewInfoFunc(fi, "glUniform2uiv", glUniform2uiv == NULL);
  glewInfoFunc(fi, "glUniform3ui", glUniform3ui == NULL);
  glewInfoFunc(fi, "glUniform3uiv", glUniform3uiv == NULL);
  glewInfoFunc(fi, "glUniform4ui", glUniform4ui == NULL);
  glewInfoFunc(fi, "glUniform4uiv", glUniform4uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribI1i", glVertexAttribI1i == NULL);
  glewInfoFunc(fi, "glVertexAttribI1iv", glVertexAttribI1iv == NULL);
  glewInfoFunc(fi, "glVertexAttribI1ui", glVertexAttribI1ui == NULL);
  glewInfoFunc(fi, "glVertexAttribI1uiv", glVertexAttribI1uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribI2i", glVertexAttribI2i == NULL);
  glewInfoFunc(fi, "glVertexAttribI2iv", glVertexAttribI2iv == NULL);
  glewInfoFunc(fi, "glVertexAttribI2ui", glVertexAttribI2ui == NULL);
  glewInfoFunc(fi, "glVertexAttribI2uiv", glVertexAttribI2uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribI3i", glVertexAttribI3i == NULL);
  glewInfoFunc(fi, "glVertexAttribI3iv", glVertexAttribI3iv == NULL);
  glewInfoFunc(fi, "glVertexAttribI3ui", glVertexAttribI3ui == NULL);
  glewInfoFunc(fi, "glVertexAttribI3uiv", glVertexAttribI3uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribI4bv", glVertexAttribI4bv == NULL);
  glewInfoFunc(fi, "glVertexAttribI4i", glVertexAttribI4i == NULL);
  glewInfoFunc(fi, "glVertexAttribI4iv", glVertexAttribI4iv == NULL);
  glewInfoFunc(fi, "glVertexAttribI4sv", glVertexAttribI4sv == NULL);
  glewInfoFunc(fi, "glVertexAttribI4ubv", glVertexAttribI4ubv == NULL);
  glewInfoFunc(fi, "glVertexAttribI4ui", glVertexAttribI4ui == NULL);
  glewInfoFunc(fi, "glVertexAttribI4uiv", glVertexAttribI4uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribI4usv", glVertexAttribI4usv == NULL);
  glewInfoFunc(fi, "glVertexAttribIPointer", glVertexAttribIPointer == NULL);
}

#endif /* GL_VERSION_3_0 */

#ifdef GL_VERSION_3_1

static void _glewInfo_GL_VERSION_3_1 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_3_1", GLEW_VERSION_3_1, GLEW_VERSION_3_1, GLEW_VERSION_3_1);

  glewInfoFunc(fi, "glDrawArraysInstanced", glDrawArraysInstanced == NULL);
  glewInfoFunc(fi, "glDrawElementsInstanced", glDrawElementsInstanced == NULL);
  glewInfoFunc(fi, "glPrimitiveRestartIndex", glPrimitiveRestartIndex == NULL);
  glewInfoFunc(fi, "glTexBuffer", glTexBuffer == NULL);
}

#endif /* GL_VERSION_3_1 */

#ifdef GL_VERSION_3_2

static void _glewInfo_GL_VERSION_3_2 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_3_2", GLEW_VERSION_3_2, GLEW_VERSION_3_2, GLEW_VERSION_3_2);

  glewInfoFunc(fi, "glFramebufferTexture", glFramebufferTexture == NULL);
  glewInfoFunc(fi, "glGetBufferParameteri64v", glGetBufferParameteri64v == NULL);
  glewInfoFunc(fi, "glGetInteger64i_v", glGetInteger64i_v == NULL);
}

#endif /* GL_VERSION_3_2 */

#ifdef GL_VERSION_3_3

static void _glewInfo_GL_VERSION_3_3 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_3_3", GLEW_VERSION_3_3, GLEW_VERSION_3_3, GLEW_VERSION_3_3);

  glewInfoFunc(fi, "glVertexAttribDivisor", glVertexAttribDivisor == NULL);
}

#endif /* GL_VERSION_3_3 */

#ifdef GL_VERSION_4_0

static void _glewInfo_GL_VERSION_4_0 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_4_0", GLEW_VERSION_4_0, GLEW_VERSION_4_0, GLEW_VERSION_4_0);

  glewInfoFunc(fi, "glBlendEquationSeparatei", glBlendEquationSeparatei == NULL);
  glewInfoFunc(fi, "glBlendEquationi", glBlendEquationi == NULL);
  glewInfoFunc(fi, "glBlendFuncSeparatei", glBlendFuncSeparatei == NULL);
  glewInfoFunc(fi, "glBlendFunci", glBlendFunci == NULL);
  glewInfoFunc(fi, "glMinSampleShading", glMinSampleShading == NULL);
}

#endif /* GL_VERSION_4_0 */

#ifdef GL_VERSION_4_1

static void _glewInfo_GL_VERSION_4_1 (void)
{
  glewPrintExt("GL_VERSION_4_1", GLEW_VERSION_4_1, GLEW_VERSION_4_1, GLEW_VERSION_4_1);
}

#endif /* GL_VERSION_4_1 */

#ifdef GL_VERSION_4_2

static void _glewInfo_GL_VERSION_4_2 (void)
{
  glewPrintExt("GL_VERSION_4_2", GLEW_VERSION_4_2, GLEW_VERSION_4_2, GLEW_VERSION_4_2);
}

#endif /* GL_VERSION_4_2 */

#ifdef GL_VERSION_4_3

static void _glewInfo_GL_VERSION_4_3 (void)
{
  glewPrintExt("GL_VERSION_4_3", GLEW_VERSION_4_3, GLEW_VERSION_4_3, GLEW_VERSION_4_3);
}

#endif /* GL_VERSION_4_3 */

#ifdef GL_VERSION_4_4

static void _glewInfo_GL_VERSION_4_4 (void)
{
  glewPrintExt("GL_VERSION_4_4", GLEW_VERSION_4_4, GLEW_VERSION_4_4, GLEW_VERSION_4_4);
}

#endif /* GL_VERSION_4_4 */

#ifdef GL_VERSION_4_5

static void _glewInfo_GL_VERSION_4_5 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_4_5", GLEW_VERSION_4_5, GLEW_VERSION_4_5, GLEW_VERSION_4_5);

  glewInfoFunc(fi, "glGetGraphicsResetStatus", glGetGraphicsResetStatus == NULL);
  glewInfoFunc(fi, "glGetnCompressedTexImage", glGetnCompressedTexImage == NULL);
  glewInfoFunc(fi, "glGetnTexImage", glGetnTexImage == NULL);
  glewInfoFunc(fi, "glGetnUniformdv", glGetnUniformdv == NULL);
}

#endif /* GL_VERSION_4_5 */

#ifdef GL_VERSION_4_6

static void _glewInfo_GL_VERSION_4_6 (void)
{
  GLboolean fi = glewPrintExt("GL_VERSION_4_6", GLEW_VERSION_4_6, GLEW_VERSION_4_6, GLEW_VERSION_4_6);

  glewInfoFunc(fi, "glMultiDrawArraysIndirectCount", glMultiDrawArraysIndirectCount == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsIndirectCount", glMultiDrawElementsIndirectCount == NULL);
  glewInfoFunc(fi, "glSpecializeShader", glSpecializeShader == NULL);
}

#endif /* GL_VERSION_4_6 */

#ifdef GL_3DFX_multisample

static void _glewInfo_GL_3DFX_multisample (void)
{
  glewPrintExt("GL_3DFX_multisample", GLEW_3DFX_multisample, glewIsSupported("GL_3DFX_multisample"), glewGetExtension("GL_3DFX_multisample"));
}

#endif /* GL_3DFX_multisample */

#ifdef GL_3DFX_tbuffer

static void _glewInfo_GL_3DFX_tbuffer (void)
{
  GLboolean fi = glewPrintExt("GL_3DFX_tbuffer", GLEW_3DFX_tbuffer, glewIsSupported("GL_3DFX_tbuffer"), glewGetExtension("GL_3DFX_tbuffer"));

  glewInfoFunc(fi, "glTbufferMask3DFX", glTbufferMask3DFX == NULL);
}

#endif /* GL_3DFX_tbuffer */

#ifdef GL_3DFX_texture_compression_FXT1

static void _glewInfo_GL_3DFX_texture_compression_FXT1 (void)
{
  glewPrintExt("GL_3DFX_texture_compression_FXT1", GLEW_3DFX_texture_compression_FXT1, glewIsSupported("GL_3DFX_texture_compression_FXT1"), glewGetExtension("GL_3DFX_texture_compression_FXT1"));
}

#endif /* GL_3DFX_texture_compression_FXT1 */

#ifdef GL_AMD_blend_minmax_factor

static void _glewInfo_GL_AMD_blend_minmax_factor (void)
{
  glewPrintExt("GL_AMD_blend_minmax_factor", GLEW_AMD_blend_minmax_factor, glewIsSupported("GL_AMD_blend_minmax_factor"), glewGetExtension("GL_AMD_blend_minmax_factor"));
}

#endif /* GL_AMD_blend_minmax_factor */

#ifdef GL_AMD_compressed_3DC_texture

static void _glewInfo_GL_AMD_compressed_3DC_texture (void)
{
  glewPrintExt("GL_AMD_compressed_3DC_texture", GLEW_AMD_compressed_3DC_texture, glewIsSupported("GL_AMD_compressed_3DC_texture"), glewGetExtension("GL_AMD_compressed_3DC_texture"));
}

#endif /* GL_AMD_compressed_3DC_texture */

#ifdef GL_AMD_compressed_ATC_texture

static void _glewInfo_GL_AMD_compressed_ATC_texture (void)
{
  glewPrintExt("GL_AMD_compressed_ATC_texture", GLEW_AMD_compressed_ATC_texture, glewIsSupported("GL_AMD_compressed_ATC_texture"), glewGetExtension("GL_AMD_compressed_ATC_texture"));
}

#endif /* GL_AMD_compressed_ATC_texture */

#ifdef GL_AMD_conservative_depth

static void _glewInfo_GL_AMD_conservative_depth (void)
{
  glewPrintExt("GL_AMD_conservative_depth", GLEW_AMD_conservative_depth, glewIsSupported("GL_AMD_conservative_depth"), glewGetExtension("GL_AMD_conservative_depth"));
}

#endif /* GL_AMD_conservative_depth */

#ifdef GL_AMD_debug_output

static void _glewInfo_GL_AMD_debug_output (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_debug_output", GLEW_AMD_debug_output, glewIsSupported("GL_AMD_debug_output"), glewGetExtension("GL_AMD_debug_output"));

  glewInfoFunc(fi, "glDebugMessageCallbackAMD", glDebugMessageCallbackAMD == NULL);
  glewInfoFunc(fi, "glDebugMessageEnableAMD", glDebugMessageEnableAMD == NULL);
  glewInfoFunc(fi, "glDebugMessageInsertAMD", glDebugMessageInsertAMD == NULL);
  glewInfoFunc(fi, "glGetDebugMessageLogAMD", glGetDebugMessageLogAMD == NULL);
}

#endif /* GL_AMD_debug_output */

#ifdef GL_AMD_depth_clamp_separate

static void _glewInfo_GL_AMD_depth_clamp_separate (void)
{
  glewPrintExt("GL_AMD_depth_clamp_separate", GLEW_AMD_depth_clamp_separate, glewIsSupported("GL_AMD_depth_clamp_separate"), glewGetExtension("GL_AMD_depth_clamp_separate"));
}

#endif /* GL_AMD_depth_clamp_separate */

#ifdef GL_AMD_draw_buffers_blend

static void _glewInfo_GL_AMD_draw_buffers_blend (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_draw_buffers_blend", GLEW_AMD_draw_buffers_blend, glewIsSupported("GL_AMD_draw_buffers_blend"), glewGetExtension("GL_AMD_draw_buffers_blend"));

  glewInfoFunc(fi, "glBlendEquationIndexedAMD", glBlendEquationIndexedAMD == NULL);
  glewInfoFunc(fi, "glBlendEquationSeparateIndexedAMD", glBlendEquationSeparateIndexedAMD == NULL);
  glewInfoFunc(fi, "glBlendFuncIndexedAMD", glBlendFuncIndexedAMD == NULL);
  glewInfoFunc(fi, "glBlendFuncSeparateIndexedAMD", glBlendFuncSeparateIndexedAMD == NULL);
}

#endif /* GL_AMD_draw_buffers_blend */

#ifdef GL_AMD_framebuffer_multisample_advanced

static void _glewInfo_GL_AMD_framebuffer_multisample_advanced (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_framebuffer_multisample_advanced", GLEW_AMD_framebuffer_multisample_advanced, glewIsSupported("GL_AMD_framebuffer_multisample_advanced"), glewGetExtension("GL_AMD_framebuffer_multisample_advanced"));

  glewInfoFunc(fi, "glNamedRenderbufferStorageMultisampleAdvancedAMD", glNamedRenderbufferStorageMultisampleAdvancedAMD == NULL);
  glewInfoFunc(fi, "glRenderbufferStorageMultisampleAdvancedAMD", glRenderbufferStorageMultisampleAdvancedAMD == NULL);
}

#endif /* GL_AMD_framebuffer_multisample_advanced */

#ifdef GL_AMD_framebuffer_sample_positions

static void _glewInfo_GL_AMD_framebuffer_sample_positions (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_framebuffer_sample_positions", GLEW_AMD_framebuffer_sample_positions, glewIsSupported("GL_AMD_framebuffer_sample_positions"), glewGetExtension("GL_AMD_framebuffer_sample_positions"));

  glewInfoFunc(fi, "glFramebufferSamplePositionsfvAMD", glFramebufferSamplePositionsfvAMD == NULL);
  glewInfoFunc(fi, "glGetFramebufferParameterfvAMD", glGetFramebufferParameterfvAMD == NULL);
  glewInfoFunc(fi, "glGetNamedFramebufferParameterfvAMD", glGetNamedFramebufferParameterfvAMD == NULL);
  glewInfoFunc(fi, "glNamedFramebufferSamplePositionsfvAMD", glNamedFramebufferSamplePositionsfvAMD == NULL);
}

#endif /* GL_AMD_framebuffer_sample_positions */

#ifdef GL_AMD_gcn_shader

static void _glewInfo_GL_AMD_gcn_shader (void)
{
  glewPrintExt("GL_AMD_gcn_shader", GLEW_AMD_gcn_shader, glewIsSupported("GL_AMD_gcn_shader"), glewGetExtension("GL_AMD_gcn_shader"));
}

#endif /* GL_AMD_gcn_shader */

#ifdef GL_AMD_gpu_shader_half_float

static void _glewInfo_GL_AMD_gpu_shader_half_float (void)
{
  glewPrintExt("GL_AMD_gpu_shader_half_float", GLEW_AMD_gpu_shader_half_float, glewIsSupported("GL_AMD_gpu_shader_half_float"), glewGetExtension("GL_AMD_gpu_shader_half_float"));
}

#endif /* GL_AMD_gpu_shader_half_float */

#ifdef GL_AMD_gpu_shader_half_float_fetch

static void _glewInfo_GL_AMD_gpu_shader_half_float_fetch (void)
{
  glewPrintExt("GL_AMD_gpu_shader_half_float_fetch", GLEW_AMD_gpu_shader_half_float_fetch, glewIsSupported("GL_AMD_gpu_shader_half_float_fetch"), glewGetExtension("GL_AMD_gpu_shader_half_float_fetch"));
}

#endif /* GL_AMD_gpu_shader_half_float_fetch */

#ifdef GL_AMD_gpu_shader_int16

static void _glewInfo_GL_AMD_gpu_shader_int16 (void)
{
  glewPrintExt("GL_AMD_gpu_shader_int16", GLEW_AMD_gpu_shader_int16, glewIsSupported("GL_AMD_gpu_shader_int16"), glewGetExtension("GL_AMD_gpu_shader_int16"));
}

#endif /* GL_AMD_gpu_shader_int16 */

#ifdef GL_AMD_gpu_shader_int64

static void _glewInfo_GL_AMD_gpu_shader_int64 (void)
{
  glewPrintExt("GL_AMD_gpu_shader_int64", GLEW_AMD_gpu_shader_int64, glewIsSupported("GL_AMD_gpu_shader_int64"), glewGetExtension("GL_AMD_gpu_shader_int64"));
}

#endif /* GL_AMD_gpu_shader_int64 */

#ifdef GL_AMD_interleaved_elements

static void _glewInfo_GL_AMD_interleaved_elements (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_interleaved_elements", GLEW_AMD_interleaved_elements, glewIsSupported("GL_AMD_interleaved_elements"), glewGetExtension("GL_AMD_interleaved_elements"));

  glewInfoFunc(fi, "glVertexAttribParameteriAMD", glVertexAttribParameteriAMD == NULL);
}

#endif /* GL_AMD_interleaved_elements */

#ifdef GL_AMD_multi_draw_indirect

static void _glewInfo_GL_AMD_multi_draw_indirect (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_multi_draw_indirect", GLEW_AMD_multi_draw_indirect, glewIsSupported("GL_AMD_multi_draw_indirect"), glewGetExtension("GL_AMD_multi_draw_indirect"));

  glewInfoFunc(fi, "glMultiDrawArraysIndirectAMD", glMultiDrawArraysIndirectAMD == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsIndirectAMD", glMultiDrawElementsIndirectAMD == NULL);
}

#endif /* GL_AMD_multi_draw_indirect */

#ifdef GL_AMD_name_gen_delete

static void _glewInfo_GL_AMD_name_gen_delete (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_name_gen_delete", GLEW_AMD_name_gen_delete, glewIsSupported("GL_AMD_name_gen_delete"), glewGetExtension("GL_AMD_name_gen_delete"));

  glewInfoFunc(fi, "glDeleteNamesAMD", glDeleteNamesAMD == NULL);
  glewInfoFunc(fi, "glGenNamesAMD", glGenNamesAMD == NULL);
  glewInfoFunc(fi, "glIsNameAMD", glIsNameAMD == NULL);
}

#endif /* GL_AMD_name_gen_delete */

#ifdef GL_AMD_occlusion_query_event

static void _glewInfo_GL_AMD_occlusion_query_event (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_occlusion_query_event", GLEW_AMD_occlusion_query_event, glewIsSupported("GL_AMD_occlusion_query_event"), glewGetExtension("GL_AMD_occlusion_query_event"));

  glewInfoFunc(fi, "glQueryObjectParameteruiAMD", glQueryObjectParameteruiAMD == NULL);
}

#endif /* GL_AMD_occlusion_query_event */

#ifdef GL_AMD_performance_monitor

static void _glewInfo_GL_AMD_performance_monitor (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_performance_monitor", GLEW_AMD_performance_monitor, glewIsSupported("GL_AMD_performance_monitor"), glewGetExtension("GL_AMD_performance_monitor"));

  glewInfoFunc(fi, "glBeginPerfMonitorAMD", glBeginPerfMonitorAMD == NULL);
  glewInfoFunc(fi, "glDeletePerfMonitorsAMD", glDeletePerfMonitorsAMD == NULL);
  glewInfoFunc(fi, "glEndPerfMonitorAMD", glEndPerfMonitorAMD == NULL);
  glewInfoFunc(fi, "glGenPerfMonitorsAMD", glGenPerfMonitorsAMD == NULL);
  glewInfoFunc(fi, "glGetPerfMonitorCounterDataAMD", glGetPerfMonitorCounterDataAMD == NULL);
  glewInfoFunc(fi, "glGetPerfMonitorCounterInfoAMD", glGetPerfMonitorCounterInfoAMD == NULL);
  glewInfoFunc(fi, "glGetPerfMonitorCounterStringAMD", glGetPerfMonitorCounterStringAMD == NULL);
  glewInfoFunc(fi, "glGetPerfMonitorCountersAMD", glGetPerfMonitorCountersAMD == NULL);
  glewInfoFunc(fi, "glGetPerfMonitorGroupStringAMD", glGetPerfMonitorGroupStringAMD == NULL);
  glewInfoFunc(fi, "glGetPerfMonitorGroupsAMD", glGetPerfMonitorGroupsAMD == NULL);
  glewInfoFunc(fi, "glSelectPerfMonitorCountersAMD", glSelectPerfMonitorCountersAMD == NULL);
}

#endif /* GL_AMD_performance_monitor */

#ifdef GL_AMD_pinned_memory

static void _glewInfo_GL_AMD_pinned_memory (void)
{
  glewPrintExt("GL_AMD_pinned_memory", GLEW_AMD_pinned_memory, glewIsSupported("GL_AMD_pinned_memory"), glewGetExtension("GL_AMD_pinned_memory"));
}

#endif /* GL_AMD_pinned_memory */

#ifdef GL_AMD_program_binary_Z400

static void _glewInfo_GL_AMD_program_binary_Z400 (void)
{
  glewPrintExt("GL_AMD_program_binary_Z400", GLEW_AMD_program_binary_Z400, glewIsSupported("GL_AMD_program_binary_Z400"), glewGetExtension("GL_AMD_program_binary_Z400"));
}

#endif /* GL_AMD_program_binary_Z400 */

#ifdef GL_AMD_query_buffer_object

static void _glewInfo_GL_AMD_query_buffer_object (void)
{
  glewPrintExt("GL_AMD_query_buffer_object", GLEW_AMD_query_buffer_object, glewIsSupported("GL_AMD_query_buffer_object"), glewGetExtension("GL_AMD_query_buffer_object"));
}

#endif /* GL_AMD_query_buffer_object */

#ifdef GL_AMD_sample_positions

static void _glewInfo_GL_AMD_sample_positions (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_sample_positions", GLEW_AMD_sample_positions, glewIsSupported("GL_AMD_sample_positions"), glewGetExtension("GL_AMD_sample_positions"));

  glewInfoFunc(fi, "glSetMultisamplefvAMD", glSetMultisamplefvAMD == NULL);
}

#endif /* GL_AMD_sample_positions */

#ifdef GL_AMD_seamless_cubemap_per_texture

static void _glewInfo_GL_AMD_seamless_cubemap_per_texture (void)
{
  glewPrintExt("GL_AMD_seamless_cubemap_per_texture", GLEW_AMD_seamless_cubemap_per_texture, glewIsSupported("GL_AMD_seamless_cubemap_per_texture"), glewGetExtension("GL_AMD_seamless_cubemap_per_texture"));
}

#endif /* GL_AMD_seamless_cubemap_per_texture */

#ifdef GL_AMD_shader_atomic_counter_ops

static void _glewInfo_GL_AMD_shader_atomic_counter_ops (void)
{
  glewPrintExt("GL_AMD_shader_atomic_counter_ops", GLEW_AMD_shader_atomic_counter_ops, glewIsSupported("GL_AMD_shader_atomic_counter_ops"), glewGetExtension("GL_AMD_shader_atomic_counter_ops"));
}

#endif /* GL_AMD_shader_atomic_counter_ops */

#ifdef GL_AMD_shader_ballot

static void _glewInfo_GL_AMD_shader_ballot (void)
{
  glewPrintExt("GL_AMD_shader_ballot", GLEW_AMD_shader_ballot, glewIsSupported("GL_AMD_shader_ballot"), glewGetExtension("GL_AMD_shader_ballot"));
}

#endif /* GL_AMD_shader_ballot */

#ifdef GL_AMD_shader_explicit_vertex_parameter

static void _glewInfo_GL_AMD_shader_explicit_vertex_parameter (void)
{
  glewPrintExt("GL_AMD_shader_explicit_vertex_parameter", GLEW_AMD_shader_explicit_vertex_parameter, glewIsSupported("GL_AMD_shader_explicit_vertex_parameter"), glewGetExtension("GL_AMD_shader_explicit_vertex_parameter"));
}

#endif /* GL_AMD_shader_explicit_vertex_parameter */

#ifdef GL_AMD_shader_image_load_store_lod

static void _glewInfo_GL_AMD_shader_image_load_store_lod (void)
{
  glewPrintExt("GL_AMD_shader_image_load_store_lod", GLEW_AMD_shader_image_load_store_lod, glewIsSupported("GL_AMD_shader_image_load_store_lod"), glewGetExtension("GL_AMD_shader_image_load_store_lod"));
}

#endif /* GL_AMD_shader_image_load_store_lod */

#ifdef GL_AMD_shader_stencil_export

static void _glewInfo_GL_AMD_shader_stencil_export (void)
{
  glewPrintExt("GL_AMD_shader_stencil_export", GLEW_AMD_shader_stencil_export, glewIsSupported("GL_AMD_shader_stencil_export"), glewGetExtension("GL_AMD_shader_stencil_export"));
}

#endif /* GL_AMD_shader_stencil_export */

#ifdef GL_AMD_shader_stencil_value_export

static void _glewInfo_GL_AMD_shader_stencil_value_export (void)
{
  glewPrintExt("GL_AMD_shader_stencil_value_export", GLEW_AMD_shader_stencil_value_export, glewIsSupported("GL_AMD_shader_stencil_value_export"), glewGetExtension("GL_AMD_shader_stencil_value_export"));
}

#endif /* GL_AMD_shader_stencil_value_export */

#ifdef GL_AMD_shader_trinary_minmax

static void _glewInfo_GL_AMD_shader_trinary_minmax (void)
{
  glewPrintExt("GL_AMD_shader_trinary_minmax", GLEW_AMD_shader_trinary_minmax, glewIsSupported("GL_AMD_shader_trinary_minmax"), glewGetExtension("GL_AMD_shader_trinary_minmax"));
}

#endif /* GL_AMD_shader_trinary_minmax */

#ifdef GL_AMD_sparse_texture

static void _glewInfo_GL_AMD_sparse_texture (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_sparse_texture", GLEW_AMD_sparse_texture, glewIsSupported("GL_AMD_sparse_texture"), glewGetExtension("GL_AMD_sparse_texture"));

  glewInfoFunc(fi, "glTexStorageSparseAMD", glTexStorageSparseAMD == NULL);
  glewInfoFunc(fi, "glTextureStorageSparseAMD", glTextureStorageSparseAMD == NULL);
}

#endif /* GL_AMD_sparse_texture */

#ifdef GL_AMD_stencil_operation_extended

static void _glewInfo_GL_AMD_stencil_operation_extended (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_stencil_operation_extended", GLEW_AMD_stencil_operation_extended, glewIsSupported("GL_AMD_stencil_operation_extended"), glewGetExtension("GL_AMD_stencil_operation_extended"));

  glewInfoFunc(fi, "glStencilOpValueAMD", glStencilOpValueAMD == NULL);
}

#endif /* GL_AMD_stencil_operation_extended */

#ifdef GL_AMD_texture_gather_bias_lod

static void _glewInfo_GL_AMD_texture_gather_bias_lod (void)
{
  glewPrintExt("GL_AMD_texture_gather_bias_lod", GLEW_AMD_texture_gather_bias_lod, glewIsSupported("GL_AMD_texture_gather_bias_lod"), glewGetExtension("GL_AMD_texture_gather_bias_lod"));
}

#endif /* GL_AMD_texture_gather_bias_lod */

#ifdef GL_AMD_texture_texture4

static void _glewInfo_GL_AMD_texture_texture4 (void)
{
  glewPrintExt("GL_AMD_texture_texture4", GLEW_AMD_texture_texture4, glewIsSupported("GL_AMD_texture_texture4"), glewGetExtension("GL_AMD_texture_texture4"));
}

#endif /* GL_AMD_texture_texture4 */

#ifdef GL_AMD_transform_feedback3_lines_triangles

static void _glewInfo_GL_AMD_transform_feedback3_lines_triangles (void)
{
  glewPrintExt("GL_AMD_transform_feedback3_lines_triangles", GLEW_AMD_transform_feedback3_lines_triangles, glewIsSupported("GL_AMD_transform_feedback3_lines_triangles"), glewGetExtension("GL_AMD_transform_feedback3_lines_triangles"));
}

#endif /* GL_AMD_transform_feedback3_lines_triangles */

#ifdef GL_AMD_transform_feedback4

static void _glewInfo_GL_AMD_transform_feedback4 (void)
{
  glewPrintExt("GL_AMD_transform_feedback4", GLEW_AMD_transform_feedback4, glewIsSupported("GL_AMD_transform_feedback4"), glewGetExtension("GL_AMD_transform_feedback4"));
}

#endif /* GL_AMD_transform_feedback4 */

#ifdef GL_AMD_vertex_shader_layer

static void _glewInfo_GL_AMD_vertex_shader_layer (void)
{
  glewPrintExt("GL_AMD_vertex_shader_layer", GLEW_AMD_vertex_shader_layer, glewIsSupported("GL_AMD_vertex_shader_layer"), glewGetExtension("GL_AMD_vertex_shader_layer"));
}

#endif /* GL_AMD_vertex_shader_layer */

#ifdef GL_AMD_vertex_shader_tessellator

static void _glewInfo_GL_AMD_vertex_shader_tessellator (void)
{
  GLboolean fi = glewPrintExt("GL_AMD_vertex_shader_tessellator", GLEW_AMD_vertex_shader_tessellator, glewIsSupported("GL_AMD_vertex_shader_tessellator"), glewGetExtension("GL_AMD_vertex_shader_tessellator"));

  glewInfoFunc(fi, "glTessellationFactorAMD", glTessellationFactorAMD == NULL);
  glewInfoFunc(fi, "glTessellationModeAMD", glTessellationModeAMD == NULL);
}

#endif /* GL_AMD_vertex_shader_tessellator */

#ifdef GL_AMD_vertex_shader_viewport_index

static void _glewInfo_GL_AMD_vertex_shader_viewport_index (void)
{
  glewPrintExt("GL_AMD_vertex_shader_viewport_index", GLEW_AMD_vertex_shader_viewport_index, glewIsSupported("GL_AMD_vertex_shader_viewport_index"), glewGetExtension("GL_AMD_vertex_shader_viewport_index"));
}

#endif /* GL_AMD_vertex_shader_viewport_index */

#ifdef GL_ANDROID_extension_pack_es31a

static void _glewInfo_GL_ANDROID_extension_pack_es31a (void)
{
  glewPrintExt("GL_ANDROID_extension_pack_es31a", GLEW_ANDROID_extension_pack_es31a, glewIsSupported("GL_ANDROID_extension_pack_es31a"), glewGetExtension("GL_ANDROID_extension_pack_es31a"));
}

#endif /* GL_ANDROID_extension_pack_es31a */

#ifdef GL_ANGLE_depth_texture

static void _glewInfo_GL_ANGLE_depth_texture (void)
{
  glewPrintExt("GL_ANGLE_depth_texture", GLEW_ANGLE_depth_texture, glewIsSupported("GL_ANGLE_depth_texture"), glewGetExtension("GL_ANGLE_depth_texture"));
}

#endif /* GL_ANGLE_depth_texture */

#ifdef GL_ANGLE_framebuffer_blit

static void _glewInfo_GL_ANGLE_framebuffer_blit (void)
{
  GLboolean fi = glewPrintExt("GL_ANGLE_framebuffer_blit", GLEW_ANGLE_framebuffer_blit, glewIsSupported("GL_ANGLE_framebuffer_blit"), glewGetExtension("GL_ANGLE_framebuffer_blit"));

  glewInfoFunc(fi, "glBlitFramebufferANGLE", glBlitFramebufferANGLE == NULL);
}

#endif /* GL_ANGLE_framebuffer_blit */

#ifdef GL_ANGLE_framebuffer_multisample

static void _glewInfo_GL_ANGLE_framebuffer_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_ANGLE_framebuffer_multisample", GLEW_ANGLE_framebuffer_multisample, glewIsSupported("GL_ANGLE_framebuffer_multisample"), glewGetExtension("GL_ANGLE_framebuffer_multisample"));

  glewInfoFunc(fi, "glRenderbufferStorageMultisampleANGLE", glRenderbufferStorageMultisampleANGLE == NULL);
}

#endif /* GL_ANGLE_framebuffer_multisample */

#ifdef GL_ANGLE_instanced_arrays

static void _glewInfo_GL_ANGLE_instanced_arrays (void)
{
  GLboolean fi = glewPrintExt("GL_ANGLE_instanced_arrays", GLEW_ANGLE_instanced_arrays, glewIsSupported("GL_ANGLE_instanced_arrays"), glewGetExtension("GL_ANGLE_instanced_arrays"));

  glewInfoFunc(fi, "glDrawArraysInstancedANGLE", glDrawArraysInstancedANGLE == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedANGLE", glDrawElementsInstancedANGLE == NULL);
  glewInfoFunc(fi, "glVertexAttribDivisorANGLE", glVertexAttribDivisorANGLE == NULL);
}

#endif /* GL_ANGLE_instanced_arrays */

#ifdef GL_ANGLE_pack_reverse_row_order

static void _glewInfo_GL_ANGLE_pack_reverse_row_order (void)
{
  glewPrintExt("GL_ANGLE_pack_reverse_row_order", GLEW_ANGLE_pack_reverse_row_order, glewIsSupported("GL_ANGLE_pack_reverse_row_order"), glewGetExtension("GL_ANGLE_pack_reverse_row_order"));
}

#endif /* GL_ANGLE_pack_reverse_row_order */

#ifdef GL_ANGLE_program_binary

static void _glewInfo_GL_ANGLE_program_binary (void)
{
  glewPrintExt("GL_ANGLE_program_binary", GLEW_ANGLE_program_binary, glewIsSupported("GL_ANGLE_program_binary"), glewGetExtension("GL_ANGLE_program_binary"));
}

#endif /* GL_ANGLE_program_binary */

#ifdef GL_ANGLE_texture_compression_dxt1

static void _glewInfo_GL_ANGLE_texture_compression_dxt1 (void)
{
  glewPrintExt("GL_ANGLE_texture_compression_dxt1", GLEW_ANGLE_texture_compression_dxt1, glewIsSupported("GL_ANGLE_texture_compression_dxt1"), glewGetExtension("GL_ANGLE_texture_compression_dxt1"));
}

#endif /* GL_ANGLE_texture_compression_dxt1 */

#ifdef GL_ANGLE_texture_compression_dxt3

static void _glewInfo_GL_ANGLE_texture_compression_dxt3 (void)
{
  glewPrintExt("GL_ANGLE_texture_compression_dxt3", GLEW_ANGLE_texture_compression_dxt3, glewIsSupported("GL_ANGLE_texture_compression_dxt3"), glewGetExtension("GL_ANGLE_texture_compression_dxt3"));
}

#endif /* GL_ANGLE_texture_compression_dxt3 */

#ifdef GL_ANGLE_texture_compression_dxt5

static void _glewInfo_GL_ANGLE_texture_compression_dxt5 (void)
{
  glewPrintExt("GL_ANGLE_texture_compression_dxt5", GLEW_ANGLE_texture_compression_dxt5, glewIsSupported("GL_ANGLE_texture_compression_dxt5"), glewGetExtension("GL_ANGLE_texture_compression_dxt5"));
}

#endif /* GL_ANGLE_texture_compression_dxt5 */

#ifdef GL_ANGLE_texture_usage

static void _glewInfo_GL_ANGLE_texture_usage (void)
{
  glewPrintExt("GL_ANGLE_texture_usage", GLEW_ANGLE_texture_usage, glewIsSupported("GL_ANGLE_texture_usage"), glewGetExtension("GL_ANGLE_texture_usage"));
}

#endif /* GL_ANGLE_texture_usage */

#ifdef GL_ANGLE_timer_query

static void _glewInfo_GL_ANGLE_timer_query (void)
{
  GLboolean fi = glewPrintExt("GL_ANGLE_timer_query", GLEW_ANGLE_timer_query, glewIsSupported("GL_ANGLE_timer_query"), glewGetExtension("GL_ANGLE_timer_query"));

  glewInfoFunc(fi, "glBeginQueryANGLE", glBeginQueryANGLE == NULL);
  glewInfoFunc(fi, "glDeleteQueriesANGLE", glDeleteQueriesANGLE == NULL);
  glewInfoFunc(fi, "glEndQueryANGLE", glEndQueryANGLE == NULL);
  glewInfoFunc(fi, "glGenQueriesANGLE", glGenQueriesANGLE == NULL);
  glewInfoFunc(fi, "glGetQueryObjecti64vANGLE", glGetQueryObjecti64vANGLE == NULL);
  glewInfoFunc(fi, "glGetQueryObjectivANGLE", glGetQueryObjectivANGLE == NULL);
  glewInfoFunc(fi, "glGetQueryObjectui64vANGLE", glGetQueryObjectui64vANGLE == NULL);
  glewInfoFunc(fi, "glGetQueryObjectuivANGLE", glGetQueryObjectuivANGLE == NULL);
  glewInfoFunc(fi, "glGetQueryivANGLE", glGetQueryivANGLE == NULL);
  glewInfoFunc(fi, "glIsQueryANGLE", glIsQueryANGLE == NULL);
  glewInfoFunc(fi, "glQueryCounterANGLE", glQueryCounterANGLE == NULL);
}

#endif /* GL_ANGLE_timer_query */

#ifdef GL_ANGLE_translated_shader_source

static void _glewInfo_GL_ANGLE_translated_shader_source (void)
{
  GLboolean fi = glewPrintExt("GL_ANGLE_translated_shader_source", GLEW_ANGLE_translated_shader_source, glewIsSupported("GL_ANGLE_translated_shader_source"), glewGetExtension("GL_ANGLE_translated_shader_source"));

  glewInfoFunc(fi, "glGetTranslatedShaderSourceANGLE", glGetTranslatedShaderSourceANGLE == NULL);
}

#endif /* GL_ANGLE_translated_shader_source */

#ifdef GL_APPLE_aux_depth_stencil

static void _glewInfo_GL_APPLE_aux_depth_stencil (void)
{
  glewPrintExt("GL_APPLE_aux_depth_stencil", GLEW_APPLE_aux_depth_stencil, glewIsSupported("GL_APPLE_aux_depth_stencil"), glewGetExtension("GL_APPLE_aux_depth_stencil"));
}

#endif /* GL_APPLE_aux_depth_stencil */

#ifdef GL_APPLE_client_storage

static void _glewInfo_GL_APPLE_client_storage (void)
{
  glewPrintExt("GL_APPLE_client_storage", GLEW_APPLE_client_storage, glewIsSupported("GL_APPLE_client_storage"), glewGetExtension("GL_APPLE_client_storage"));
}

#endif /* GL_APPLE_client_storage */

#ifdef GL_APPLE_clip_distance

static void _glewInfo_GL_APPLE_clip_distance (void)
{
  glewPrintExt("GL_APPLE_clip_distance", GLEW_APPLE_clip_distance, glewIsSupported("GL_APPLE_clip_distance"), glewGetExtension("GL_APPLE_clip_distance"));
}

#endif /* GL_APPLE_clip_distance */

#ifdef GL_APPLE_color_buffer_packed_float

static void _glewInfo_GL_APPLE_color_buffer_packed_float (void)
{
  glewPrintExt("GL_APPLE_color_buffer_packed_float", GLEW_APPLE_color_buffer_packed_float, glewIsSupported("GL_APPLE_color_buffer_packed_float"), glewGetExtension("GL_APPLE_color_buffer_packed_float"));
}

#endif /* GL_APPLE_color_buffer_packed_float */

#ifdef GL_APPLE_copy_texture_levels

static void _glewInfo_GL_APPLE_copy_texture_levels (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_copy_texture_levels", GLEW_APPLE_copy_texture_levels, glewIsSupported("GL_APPLE_copy_texture_levels"), glewGetExtension("GL_APPLE_copy_texture_levels"));

  glewInfoFunc(fi, "glCopyTextureLevelsAPPLE", glCopyTextureLevelsAPPLE == NULL);
}

#endif /* GL_APPLE_copy_texture_levels */

#ifdef GL_APPLE_element_array

static void _glewInfo_GL_APPLE_element_array (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_element_array", GLEW_APPLE_element_array, glewIsSupported("GL_APPLE_element_array"), glewGetExtension("GL_APPLE_element_array"));

  glewInfoFunc(fi, "glDrawElementArrayAPPLE", glDrawElementArrayAPPLE == NULL);
  glewInfoFunc(fi, "glDrawRangeElementArrayAPPLE", glDrawRangeElementArrayAPPLE == NULL);
  glewInfoFunc(fi, "glElementPointerAPPLE", glElementPointerAPPLE == NULL);
  glewInfoFunc(fi, "glMultiDrawElementArrayAPPLE", glMultiDrawElementArrayAPPLE == NULL);
  glewInfoFunc(fi, "glMultiDrawRangeElementArrayAPPLE", glMultiDrawRangeElementArrayAPPLE == NULL);
}

#endif /* GL_APPLE_element_array */

#ifdef GL_APPLE_fence

static void _glewInfo_GL_APPLE_fence (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_fence", GLEW_APPLE_fence, glewIsSupported("GL_APPLE_fence"), glewGetExtension("GL_APPLE_fence"));

  glewInfoFunc(fi, "glDeleteFencesAPPLE", glDeleteFencesAPPLE == NULL);
  glewInfoFunc(fi, "glFinishFenceAPPLE", glFinishFenceAPPLE == NULL);
  glewInfoFunc(fi, "glFinishObjectAPPLE", glFinishObjectAPPLE == NULL);
  glewInfoFunc(fi, "glGenFencesAPPLE", glGenFencesAPPLE == NULL);
  glewInfoFunc(fi, "glIsFenceAPPLE", glIsFenceAPPLE == NULL);
  glewInfoFunc(fi, "glSetFenceAPPLE", glSetFenceAPPLE == NULL);
  glewInfoFunc(fi, "glTestFenceAPPLE", glTestFenceAPPLE == NULL);
  glewInfoFunc(fi, "glTestObjectAPPLE", glTestObjectAPPLE == NULL);
}

#endif /* GL_APPLE_fence */

#ifdef GL_APPLE_float_pixels

static void _glewInfo_GL_APPLE_float_pixels (void)
{
  glewPrintExt("GL_APPLE_float_pixels", GLEW_APPLE_float_pixels, glewIsSupported("GL_APPLE_float_pixels"), glewGetExtension("GL_APPLE_float_pixels"));
}

#endif /* GL_APPLE_float_pixels */

#ifdef GL_APPLE_flush_buffer_range

static void _glewInfo_GL_APPLE_flush_buffer_range (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_flush_buffer_range", GLEW_APPLE_flush_buffer_range, glewIsSupported("GL_APPLE_flush_buffer_range"), glewGetExtension("GL_APPLE_flush_buffer_range"));

  glewInfoFunc(fi, "glBufferParameteriAPPLE", glBufferParameteriAPPLE == NULL);
  glewInfoFunc(fi, "glFlushMappedBufferRangeAPPLE", glFlushMappedBufferRangeAPPLE == NULL);
}

#endif /* GL_APPLE_flush_buffer_range */

#ifdef GL_APPLE_framebuffer_multisample

static void _glewInfo_GL_APPLE_framebuffer_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_framebuffer_multisample", GLEW_APPLE_framebuffer_multisample, glewIsSupported("GL_APPLE_framebuffer_multisample"), glewGetExtension("GL_APPLE_framebuffer_multisample"));

  glewInfoFunc(fi, "glRenderbufferStorageMultisampleAPPLE", glRenderbufferStorageMultisampleAPPLE == NULL);
  glewInfoFunc(fi, "glResolveMultisampleFramebufferAPPLE", glResolveMultisampleFramebufferAPPLE == NULL);
}

#endif /* GL_APPLE_framebuffer_multisample */

#ifdef GL_APPLE_object_purgeable

static void _glewInfo_GL_APPLE_object_purgeable (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_object_purgeable", GLEW_APPLE_object_purgeable, glewIsSupported("GL_APPLE_object_purgeable"), glewGetExtension("GL_APPLE_object_purgeable"));

  glewInfoFunc(fi, "glGetObjectParameterivAPPLE", glGetObjectParameterivAPPLE == NULL);
  glewInfoFunc(fi, "glObjectPurgeableAPPLE", glObjectPurgeableAPPLE == NULL);
  glewInfoFunc(fi, "glObjectUnpurgeableAPPLE", glObjectUnpurgeableAPPLE == NULL);
}

#endif /* GL_APPLE_object_purgeable */

#ifdef GL_APPLE_pixel_buffer

static void _glewInfo_GL_APPLE_pixel_buffer (void)
{
  glewPrintExt("GL_APPLE_pixel_buffer", GLEW_APPLE_pixel_buffer, glewIsSupported("GL_APPLE_pixel_buffer"), glewGetExtension("GL_APPLE_pixel_buffer"));
}

#endif /* GL_APPLE_pixel_buffer */

#ifdef GL_APPLE_rgb_422

static void _glewInfo_GL_APPLE_rgb_422 (void)
{
  glewPrintExt("GL_APPLE_rgb_422", GLEW_APPLE_rgb_422, glewIsSupported("GL_APPLE_rgb_422"), glewGetExtension("GL_APPLE_rgb_422"));
}

#endif /* GL_APPLE_rgb_422 */

#ifdef GL_APPLE_row_bytes

static void _glewInfo_GL_APPLE_row_bytes (void)
{
  glewPrintExt("GL_APPLE_row_bytes", GLEW_APPLE_row_bytes, glewIsSupported("GL_APPLE_row_bytes"), glewGetExtension("GL_APPLE_row_bytes"));
}

#endif /* GL_APPLE_row_bytes */

#ifdef GL_APPLE_specular_vector

static void _glewInfo_GL_APPLE_specular_vector (void)
{
  glewPrintExt("GL_APPLE_specular_vector", GLEW_APPLE_specular_vector, glewIsSupported("GL_APPLE_specular_vector"), glewGetExtension("GL_APPLE_specular_vector"));
}

#endif /* GL_APPLE_specular_vector */

#ifdef GL_APPLE_sync

static void _glewInfo_GL_APPLE_sync (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_sync", GLEW_APPLE_sync, glewIsSupported("GL_APPLE_sync"), glewGetExtension("GL_APPLE_sync"));

  glewInfoFunc(fi, "glClientWaitSyncAPPLE", glClientWaitSyncAPPLE == NULL);
  glewInfoFunc(fi, "glDeleteSyncAPPLE", glDeleteSyncAPPLE == NULL);
  glewInfoFunc(fi, "glFenceSyncAPPLE", glFenceSyncAPPLE == NULL);
  glewInfoFunc(fi, "glGetInteger64vAPPLE", glGetInteger64vAPPLE == NULL);
  glewInfoFunc(fi, "glGetSyncivAPPLE", glGetSyncivAPPLE == NULL);
  glewInfoFunc(fi, "glIsSyncAPPLE", glIsSyncAPPLE == NULL);
  glewInfoFunc(fi, "glWaitSyncAPPLE", glWaitSyncAPPLE == NULL);
}

#endif /* GL_APPLE_sync */

#ifdef GL_APPLE_texture_2D_limited_npot

static void _glewInfo_GL_APPLE_texture_2D_limited_npot (void)
{
  glewPrintExt("GL_APPLE_texture_2D_limited_npot", GLEW_APPLE_texture_2D_limited_npot, glewIsSupported("GL_APPLE_texture_2D_limited_npot"), glewGetExtension("GL_APPLE_texture_2D_limited_npot"));
}

#endif /* GL_APPLE_texture_2D_limited_npot */

#ifdef GL_APPLE_texture_format_BGRA8888

static void _glewInfo_GL_APPLE_texture_format_BGRA8888 (void)
{
  glewPrintExt("GL_APPLE_texture_format_BGRA8888", GLEW_APPLE_texture_format_BGRA8888, glewIsSupported("GL_APPLE_texture_format_BGRA8888"), glewGetExtension("GL_APPLE_texture_format_BGRA8888"));
}

#endif /* GL_APPLE_texture_format_BGRA8888 */

#ifdef GL_APPLE_texture_max_level

static void _glewInfo_GL_APPLE_texture_max_level (void)
{
  glewPrintExt("GL_APPLE_texture_max_level", GLEW_APPLE_texture_max_level, glewIsSupported("GL_APPLE_texture_max_level"), glewGetExtension("GL_APPLE_texture_max_level"));
}

#endif /* GL_APPLE_texture_max_level */

#ifdef GL_APPLE_texture_packed_float

static void _glewInfo_GL_APPLE_texture_packed_float (void)
{
  glewPrintExt("GL_APPLE_texture_packed_float", GLEW_APPLE_texture_packed_float, glewIsSupported("GL_APPLE_texture_packed_float"), glewGetExtension("GL_APPLE_texture_packed_float"));
}

#endif /* GL_APPLE_texture_packed_float */

#ifdef GL_APPLE_texture_range

static void _glewInfo_GL_APPLE_texture_range (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_texture_range", GLEW_APPLE_texture_range, glewIsSupported("GL_APPLE_texture_range"), glewGetExtension("GL_APPLE_texture_range"));

  glewInfoFunc(fi, "glGetTexParameterPointervAPPLE", glGetTexParameterPointervAPPLE == NULL);
  glewInfoFunc(fi, "glTextureRangeAPPLE", glTextureRangeAPPLE == NULL);
}

#endif /* GL_APPLE_texture_range */

#ifdef GL_APPLE_transform_hint

static void _glewInfo_GL_APPLE_transform_hint (void)
{
  glewPrintExt("GL_APPLE_transform_hint", GLEW_APPLE_transform_hint, glewIsSupported("GL_APPLE_transform_hint"), glewGetExtension("GL_APPLE_transform_hint"));
}

#endif /* GL_APPLE_transform_hint */

#ifdef GL_APPLE_vertex_array_object

static void _glewInfo_GL_APPLE_vertex_array_object (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_vertex_array_object", GLEW_APPLE_vertex_array_object, glewIsSupported("GL_APPLE_vertex_array_object"), glewGetExtension("GL_APPLE_vertex_array_object"));

  glewInfoFunc(fi, "glBindVertexArrayAPPLE", glBindVertexArrayAPPLE == NULL);
  glewInfoFunc(fi, "glDeleteVertexArraysAPPLE", glDeleteVertexArraysAPPLE == NULL);
  glewInfoFunc(fi, "glGenVertexArraysAPPLE", glGenVertexArraysAPPLE == NULL);
  glewInfoFunc(fi, "glIsVertexArrayAPPLE", glIsVertexArrayAPPLE == NULL);
}

#endif /* GL_APPLE_vertex_array_object */

#ifdef GL_APPLE_vertex_array_range

static void _glewInfo_GL_APPLE_vertex_array_range (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_vertex_array_range", GLEW_APPLE_vertex_array_range, glewIsSupported("GL_APPLE_vertex_array_range"), glewGetExtension("GL_APPLE_vertex_array_range"));

  glewInfoFunc(fi, "glFlushVertexArrayRangeAPPLE", glFlushVertexArrayRangeAPPLE == NULL);
  glewInfoFunc(fi, "glVertexArrayParameteriAPPLE", glVertexArrayParameteriAPPLE == NULL);
  glewInfoFunc(fi, "glVertexArrayRangeAPPLE", glVertexArrayRangeAPPLE == NULL);
}

#endif /* GL_APPLE_vertex_array_range */

#ifdef GL_APPLE_vertex_program_evaluators

static void _glewInfo_GL_APPLE_vertex_program_evaluators (void)
{
  GLboolean fi = glewPrintExt("GL_APPLE_vertex_program_evaluators", GLEW_APPLE_vertex_program_evaluators, glewIsSupported("GL_APPLE_vertex_program_evaluators"), glewGetExtension("GL_APPLE_vertex_program_evaluators"));

  glewInfoFunc(fi, "glDisableVertexAttribAPPLE", glDisableVertexAttribAPPLE == NULL);
  glewInfoFunc(fi, "glEnableVertexAttribAPPLE", glEnableVertexAttribAPPLE == NULL);
  glewInfoFunc(fi, "glIsVertexAttribEnabledAPPLE", glIsVertexAttribEnabledAPPLE == NULL);
  glewInfoFunc(fi, "glMapVertexAttrib1dAPPLE", glMapVertexAttrib1dAPPLE == NULL);
  glewInfoFunc(fi, "glMapVertexAttrib1fAPPLE", glMapVertexAttrib1fAPPLE == NULL);
  glewInfoFunc(fi, "glMapVertexAttrib2dAPPLE", glMapVertexAttrib2dAPPLE == NULL);
  glewInfoFunc(fi, "glMapVertexAttrib2fAPPLE", glMapVertexAttrib2fAPPLE == NULL);
}

#endif /* GL_APPLE_vertex_program_evaluators */

#ifdef GL_APPLE_ycbcr_422

static void _glewInfo_GL_APPLE_ycbcr_422 (void)
{
  glewPrintExt("GL_APPLE_ycbcr_422", GLEW_APPLE_ycbcr_422, glewIsSupported("GL_APPLE_ycbcr_422"), glewGetExtension("GL_APPLE_ycbcr_422"));
}

#endif /* GL_APPLE_ycbcr_422 */

#ifdef GL_ARB_ES2_compatibility

static void _glewInfo_GL_ARB_ES2_compatibility (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_ES2_compatibility", GLEW_ARB_ES2_compatibility, glewIsSupported("GL_ARB_ES2_compatibility"), glewGetExtension("GL_ARB_ES2_compatibility"));

  glewInfoFunc(fi, "glClearDepthf", glClearDepthf == NULL);
  glewInfoFunc(fi, "glDepthRangef", glDepthRangef == NULL);
  glewInfoFunc(fi, "glGetShaderPrecisionFormat", glGetShaderPrecisionFormat == NULL);
  glewInfoFunc(fi, "glReleaseShaderCompiler", glReleaseShaderCompiler == NULL);
  glewInfoFunc(fi, "glShaderBinary", glShaderBinary == NULL);
}

#endif /* GL_ARB_ES2_compatibility */

#ifdef GL_ARB_ES3_1_compatibility

static void _glewInfo_GL_ARB_ES3_1_compatibility (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_ES3_1_compatibility", GLEW_ARB_ES3_1_compatibility, glewIsSupported("GL_ARB_ES3_1_compatibility"), glewGetExtension("GL_ARB_ES3_1_compatibility"));

  glewInfoFunc(fi, "glMemoryBarrierByRegion", glMemoryBarrierByRegion == NULL);
}

#endif /* GL_ARB_ES3_1_compatibility */

#ifdef GL_ARB_ES3_2_compatibility

static void _glewInfo_GL_ARB_ES3_2_compatibility (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_ES3_2_compatibility", GLEW_ARB_ES3_2_compatibility, glewIsSupported("GL_ARB_ES3_2_compatibility"), glewGetExtension("GL_ARB_ES3_2_compatibility"));

  glewInfoFunc(fi, "glPrimitiveBoundingBoxARB", glPrimitiveBoundingBoxARB == NULL);
}

#endif /* GL_ARB_ES3_2_compatibility */

#ifdef GL_ARB_ES3_compatibility

static void _glewInfo_GL_ARB_ES3_compatibility (void)
{
  glewPrintExt("GL_ARB_ES3_compatibility", GLEW_ARB_ES3_compatibility, glewIsSupported("GL_ARB_ES3_compatibility"), glewGetExtension("GL_ARB_ES3_compatibility"));
}

#endif /* GL_ARB_ES3_compatibility */

#ifdef GL_ARB_arrays_of_arrays

static void _glewInfo_GL_ARB_arrays_of_arrays (void)
{
  glewPrintExt("GL_ARB_arrays_of_arrays", GLEW_ARB_arrays_of_arrays, glewIsSupported("GL_ARB_arrays_of_arrays"), glewGetExtension("GL_ARB_arrays_of_arrays"));
}

#endif /* GL_ARB_arrays_of_arrays */

#ifdef GL_ARB_base_instance

static void _glewInfo_GL_ARB_base_instance (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_base_instance", GLEW_ARB_base_instance, glewIsSupported("GL_ARB_base_instance"), glewGetExtension("GL_ARB_base_instance"));

  glewInfoFunc(fi, "glDrawArraysInstancedBaseInstance", glDrawArraysInstancedBaseInstance == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedBaseInstance", glDrawElementsInstancedBaseInstance == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedBaseVertexBaseInstance", glDrawElementsInstancedBaseVertexBaseInstance == NULL);
}

#endif /* GL_ARB_base_instance */

#ifdef GL_ARB_bindless_texture

static void _glewInfo_GL_ARB_bindless_texture (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_bindless_texture", GLEW_ARB_bindless_texture, glewIsSupported("GL_ARB_bindless_texture"), glewGetExtension("GL_ARB_bindless_texture"));

  glewInfoFunc(fi, "glGetImageHandleARB", glGetImageHandleARB == NULL);
  glewInfoFunc(fi, "glGetTextureHandleARB", glGetTextureHandleARB == NULL);
  glewInfoFunc(fi, "glGetTextureSamplerHandleARB", glGetTextureSamplerHandleARB == NULL);
  glewInfoFunc(fi, "glGetVertexAttribLui64vARB", glGetVertexAttribLui64vARB == NULL);
  glewInfoFunc(fi, "glIsImageHandleResidentARB", glIsImageHandleResidentARB == NULL);
  glewInfoFunc(fi, "glIsTextureHandleResidentARB", glIsTextureHandleResidentARB == NULL);
  glewInfoFunc(fi, "glMakeImageHandleNonResidentARB", glMakeImageHandleNonResidentARB == NULL);
  glewInfoFunc(fi, "glMakeImageHandleResidentARB", glMakeImageHandleResidentARB == NULL);
  glewInfoFunc(fi, "glMakeTextureHandleNonResidentARB", glMakeTextureHandleNonResidentARB == NULL);
  glewInfoFunc(fi, "glMakeTextureHandleResidentARB", glMakeTextureHandleResidentARB == NULL);
  glewInfoFunc(fi, "glProgramUniformHandleui64ARB", glProgramUniformHandleui64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniformHandleui64vARB", glProgramUniformHandleui64vARB == NULL);
  glewInfoFunc(fi, "glUniformHandleui64ARB", glUniformHandleui64ARB == NULL);
  glewInfoFunc(fi, "glUniformHandleui64vARB", glUniformHandleui64vARB == NULL);
  glewInfoFunc(fi, "glVertexAttribL1ui64ARB", glVertexAttribL1ui64ARB == NULL);
  glewInfoFunc(fi, "glVertexAttribL1ui64vARB", glVertexAttribL1ui64vARB == NULL);
}

#endif /* GL_ARB_bindless_texture */

#ifdef GL_ARB_blend_func_extended

static void _glewInfo_GL_ARB_blend_func_extended (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_blend_func_extended", GLEW_ARB_blend_func_extended, glewIsSupported("GL_ARB_blend_func_extended"), glewGetExtension("GL_ARB_blend_func_extended"));

  glewInfoFunc(fi, "glBindFragDataLocationIndexed", glBindFragDataLocationIndexed == NULL);
  glewInfoFunc(fi, "glGetFragDataIndex", glGetFragDataIndex == NULL);
}

#endif /* GL_ARB_blend_func_extended */

#ifdef GL_ARB_buffer_storage

static void _glewInfo_GL_ARB_buffer_storage (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_buffer_storage", GLEW_ARB_buffer_storage, glewIsSupported("GL_ARB_buffer_storage"), glewGetExtension("GL_ARB_buffer_storage"));

  glewInfoFunc(fi, "glBufferStorage", glBufferStorage == NULL);
}

#endif /* GL_ARB_buffer_storage */

#ifdef GL_ARB_cl_event

static void _glewInfo_GL_ARB_cl_event (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_cl_event", GLEW_ARB_cl_event, glewIsSupported("GL_ARB_cl_event"), glewGetExtension("GL_ARB_cl_event"));

  glewInfoFunc(fi, "glCreateSyncFromCLeventARB", glCreateSyncFromCLeventARB == NULL);
}

#endif /* GL_ARB_cl_event */

#ifdef GL_ARB_clear_buffer_object

static void _glewInfo_GL_ARB_clear_buffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_clear_buffer_object", GLEW_ARB_clear_buffer_object, glewIsSupported("GL_ARB_clear_buffer_object"), glewGetExtension("GL_ARB_clear_buffer_object"));

  glewInfoFunc(fi, "glClearBufferData", glClearBufferData == NULL);
  glewInfoFunc(fi, "glClearBufferSubData", glClearBufferSubData == NULL);
  glewInfoFunc(fi, "glClearNamedBufferDataEXT", glClearNamedBufferDataEXT == NULL);
  glewInfoFunc(fi, "glClearNamedBufferSubDataEXT", glClearNamedBufferSubDataEXT == NULL);
}

#endif /* GL_ARB_clear_buffer_object */

#ifdef GL_ARB_clear_texture

static void _glewInfo_GL_ARB_clear_texture (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_clear_texture", GLEW_ARB_clear_texture, glewIsSupported("GL_ARB_clear_texture"), glewGetExtension("GL_ARB_clear_texture"));

  glewInfoFunc(fi, "glClearTexImage", glClearTexImage == NULL);
  glewInfoFunc(fi, "glClearTexSubImage", glClearTexSubImage == NULL);
}

#endif /* GL_ARB_clear_texture */

#ifdef GL_ARB_clip_control

static void _glewInfo_GL_ARB_clip_control (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_clip_control", GLEW_ARB_clip_control, glewIsSupported("GL_ARB_clip_control"), glewGetExtension("GL_ARB_clip_control"));

  glewInfoFunc(fi, "glClipControl", glClipControl == NULL);
}

#endif /* GL_ARB_clip_control */

#ifdef GL_ARB_color_buffer_float

static void _glewInfo_GL_ARB_color_buffer_float (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_color_buffer_float", GLEW_ARB_color_buffer_float, glewIsSupported("GL_ARB_color_buffer_float"), glewGetExtension("GL_ARB_color_buffer_float"));

  glewInfoFunc(fi, "glClampColorARB", glClampColorARB == NULL);
}

#endif /* GL_ARB_color_buffer_float */

#ifdef GL_ARB_compatibility

static void _glewInfo_GL_ARB_compatibility (void)
{
  glewPrintExt("GL_ARB_compatibility", GLEW_ARB_compatibility, glewIsSupported("GL_ARB_compatibility"), glewGetExtension("GL_ARB_compatibility"));
}

#endif /* GL_ARB_compatibility */

#ifdef GL_ARB_compressed_texture_pixel_storage

static void _glewInfo_GL_ARB_compressed_texture_pixel_storage (void)
{
  glewPrintExt("GL_ARB_compressed_texture_pixel_storage", GLEW_ARB_compressed_texture_pixel_storage, glewIsSupported("GL_ARB_compressed_texture_pixel_storage"), glewGetExtension("GL_ARB_compressed_texture_pixel_storage"));
}

#endif /* GL_ARB_compressed_texture_pixel_storage */

#ifdef GL_ARB_compute_shader

static void _glewInfo_GL_ARB_compute_shader (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_compute_shader", GLEW_ARB_compute_shader, glewIsSupported("GL_ARB_compute_shader"), glewGetExtension("GL_ARB_compute_shader"));

  glewInfoFunc(fi, "glDispatchCompute", glDispatchCompute == NULL);
  glewInfoFunc(fi, "glDispatchComputeIndirect", glDispatchComputeIndirect == NULL);
}

#endif /* GL_ARB_compute_shader */

#ifdef GL_ARB_compute_variable_group_size

static void _glewInfo_GL_ARB_compute_variable_group_size (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_compute_variable_group_size", GLEW_ARB_compute_variable_group_size, glewIsSupported("GL_ARB_compute_variable_group_size"), glewGetExtension("GL_ARB_compute_variable_group_size"));

  glewInfoFunc(fi, "glDispatchComputeGroupSizeARB", glDispatchComputeGroupSizeARB == NULL);
}

#endif /* GL_ARB_compute_variable_group_size */

#ifdef GL_ARB_conditional_render_inverted

static void _glewInfo_GL_ARB_conditional_render_inverted (void)
{
  glewPrintExt("GL_ARB_conditional_render_inverted", GLEW_ARB_conditional_render_inverted, glewIsSupported("GL_ARB_conditional_render_inverted"), glewGetExtension("GL_ARB_conditional_render_inverted"));
}

#endif /* GL_ARB_conditional_render_inverted */

#ifdef GL_ARB_conservative_depth

static void _glewInfo_GL_ARB_conservative_depth (void)
{
  glewPrintExt("GL_ARB_conservative_depth", GLEW_ARB_conservative_depth, glewIsSupported("GL_ARB_conservative_depth"), glewGetExtension("GL_ARB_conservative_depth"));
}

#endif /* GL_ARB_conservative_depth */

#ifdef GL_ARB_copy_buffer

static void _glewInfo_GL_ARB_copy_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_copy_buffer", GLEW_ARB_copy_buffer, glewIsSupported("GL_ARB_copy_buffer"), glewGetExtension("GL_ARB_copy_buffer"));

  glewInfoFunc(fi, "glCopyBufferSubData", glCopyBufferSubData == NULL);
}

#endif /* GL_ARB_copy_buffer */

#ifdef GL_ARB_copy_image

static void _glewInfo_GL_ARB_copy_image (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_copy_image", GLEW_ARB_copy_image, glewIsSupported("GL_ARB_copy_image"), glewGetExtension("GL_ARB_copy_image"));

  glewInfoFunc(fi, "glCopyImageSubData", glCopyImageSubData == NULL);
}

#endif /* GL_ARB_copy_image */

#ifdef GL_ARB_cull_distance

static void _glewInfo_GL_ARB_cull_distance (void)
{
  glewPrintExt("GL_ARB_cull_distance", GLEW_ARB_cull_distance, glewIsSupported("GL_ARB_cull_distance"), glewGetExtension("GL_ARB_cull_distance"));
}

#endif /* GL_ARB_cull_distance */

#ifdef GL_ARB_debug_output

static void _glewInfo_GL_ARB_debug_output (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_debug_output", GLEW_ARB_debug_output, glewIsSupported("GL_ARB_debug_output"), glewGetExtension("GL_ARB_debug_output"));

  glewInfoFunc(fi, "glDebugMessageCallbackARB", glDebugMessageCallbackARB == NULL);
  glewInfoFunc(fi, "glDebugMessageControlARB", glDebugMessageControlARB == NULL);
  glewInfoFunc(fi, "glDebugMessageInsertARB", glDebugMessageInsertARB == NULL);
  glewInfoFunc(fi, "glGetDebugMessageLogARB", glGetDebugMessageLogARB == NULL);
}

#endif /* GL_ARB_debug_output */

#ifdef GL_ARB_depth_buffer_float

static void _glewInfo_GL_ARB_depth_buffer_float (void)
{
  glewPrintExt("GL_ARB_depth_buffer_float", GLEW_ARB_depth_buffer_float, glewIsSupported("GL_ARB_depth_buffer_float"), glewGetExtension("GL_ARB_depth_buffer_float"));
}

#endif /* GL_ARB_depth_buffer_float */

#ifdef GL_ARB_depth_clamp

static void _glewInfo_GL_ARB_depth_clamp (void)
{
  glewPrintExt("GL_ARB_depth_clamp", GLEW_ARB_depth_clamp, glewIsSupported("GL_ARB_depth_clamp"), glewGetExtension("GL_ARB_depth_clamp"));
}

#endif /* GL_ARB_depth_clamp */

#ifdef GL_ARB_depth_texture

static void _glewInfo_GL_ARB_depth_texture (void)
{
  glewPrintExt("GL_ARB_depth_texture", GLEW_ARB_depth_texture, glewIsSupported("GL_ARB_depth_texture"), glewGetExtension("GL_ARB_depth_texture"));
}

#endif /* GL_ARB_depth_texture */

#ifdef GL_ARB_derivative_control

static void _glewInfo_GL_ARB_derivative_control (void)
{
  glewPrintExt("GL_ARB_derivative_control", GLEW_ARB_derivative_control, glewIsSupported("GL_ARB_derivative_control"), glewGetExtension("GL_ARB_derivative_control"));
}

#endif /* GL_ARB_derivative_control */

#ifdef GL_ARB_direct_state_access

static void _glewInfo_GL_ARB_direct_state_access (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_direct_state_access", GLEW_ARB_direct_state_access, glewIsSupported("GL_ARB_direct_state_access"), glewGetExtension("GL_ARB_direct_state_access"));

  glewInfoFunc(fi, "glBindTextureUnit", glBindTextureUnit == NULL);
  glewInfoFunc(fi, "glBlitNamedFramebuffer", glBlitNamedFramebuffer == NULL);
  glewInfoFunc(fi, "glCheckNamedFramebufferStatus", glCheckNamedFramebufferStatus == NULL);
  glewInfoFunc(fi, "glClearNamedBufferData", glClearNamedBufferData == NULL);
  glewInfoFunc(fi, "glClearNamedBufferSubData", glClearNamedBufferSubData == NULL);
  glewInfoFunc(fi, "glClearNamedFramebufferfi", glClearNamedFramebufferfi == NULL);
  glewInfoFunc(fi, "glClearNamedFramebufferfv", glClearNamedFramebufferfv == NULL);
  glewInfoFunc(fi, "glClearNamedFramebufferiv", glClearNamedFramebufferiv == NULL);
  glewInfoFunc(fi, "glClearNamedFramebufferuiv", glClearNamedFramebufferuiv == NULL);
  glewInfoFunc(fi, "glCompressedTextureSubImage1D", glCompressedTextureSubImage1D == NULL);
  glewInfoFunc(fi, "glCompressedTextureSubImage2D", glCompressedTextureSubImage2D == NULL);
  glewInfoFunc(fi, "glCompressedTextureSubImage3D", glCompressedTextureSubImage3D == NULL);
  glewInfoFunc(fi, "glCopyNamedBufferSubData", glCopyNamedBufferSubData == NULL);
  glewInfoFunc(fi, "glCopyTextureSubImage1D", glCopyTextureSubImage1D == NULL);
  glewInfoFunc(fi, "glCopyTextureSubImage2D", glCopyTextureSubImage2D == NULL);
  glewInfoFunc(fi, "glCopyTextureSubImage3D", glCopyTextureSubImage3D == NULL);
  glewInfoFunc(fi, "glCreateBuffers", glCreateBuffers == NULL);
  glewInfoFunc(fi, "glCreateFramebuffers", glCreateFramebuffers == NULL);
  glewInfoFunc(fi, "glCreateProgramPipelines", glCreateProgramPipelines == NULL);
  glewInfoFunc(fi, "glCreateQueries", glCreateQueries == NULL);
  glewInfoFunc(fi, "glCreateRenderbuffers", glCreateRenderbuffers == NULL);
  glewInfoFunc(fi, "glCreateSamplers", glCreateSamplers == NULL);
  glewInfoFunc(fi, "glCreateTextures", glCreateTextures == NULL);
  glewInfoFunc(fi, "glCreateTransformFeedbacks", glCreateTransformFeedbacks == NULL);
  glewInfoFunc(fi, "glCreateVertexArrays", glCreateVertexArrays == NULL);
  glewInfoFunc(fi, "glDisableVertexArrayAttrib", glDisableVertexArrayAttrib == NULL);
  glewInfoFunc(fi, "glEnableVertexArrayAttrib", glEnableVertexArrayAttrib == NULL);
  glewInfoFunc(fi, "glFlushMappedNamedBufferRange", glFlushMappedNamedBufferRange == NULL);
  glewInfoFunc(fi, "glGenerateTextureMipmap", glGenerateTextureMipmap == NULL);
  glewInfoFunc(fi, "glGetCompressedTextureImage", glGetCompressedTextureImage == NULL);
  glewInfoFunc(fi, "glGetNamedBufferParameteri64v", glGetNamedBufferParameteri64v == NULL);
  glewInfoFunc(fi, "glGetNamedBufferParameteriv", glGetNamedBufferParameteriv == NULL);
  glewInfoFunc(fi, "glGetNamedBufferPointerv", glGetNamedBufferPointerv == NULL);
  glewInfoFunc(fi, "glGetNamedBufferSubData", glGetNamedBufferSubData == NULL);
  glewInfoFunc(fi, "glGetNamedFramebufferAttachmentParameteriv", glGetNamedFramebufferAttachmentParameteriv == NULL);
  glewInfoFunc(fi, "glGetNamedFramebufferParameteriv", glGetNamedFramebufferParameteriv == NULL);
  glewInfoFunc(fi, "glGetNamedRenderbufferParameteriv", glGetNamedRenderbufferParameteriv == NULL);
  glewInfoFunc(fi, "glGetQueryBufferObjecti64v", glGetQueryBufferObjecti64v == NULL);
  glewInfoFunc(fi, "glGetQueryBufferObjectiv", glGetQueryBufferObjectiv == NULL);
  glewInfoFunc(fi, "glGetQueryBufferObjectui64v", glGetQueryBufferObjectui64v == NULL);
  glewInfoFunc(fi, "glGetQueryBufferObjectuiv", glGetQueryBufferObjectuiv == NULL);
  glewInfoFunc(fi, "glGetTextureImage", glGetTextureImage == NULL);
  glewInfoFunc(fi, "glGetTextureLevelParameterfv", glGetTextureLevelParameterfv == NULL);
  glewInfoFunc(fi, "glGetTextureLevelParameteriv", glGetTextureLevelParameteriv == NULL);
  glewInfoFunc(fi, "glGetTextureParameterIiv", glGetTextureParameterIiv == NULL);
  glewInfoFunc(fi, "glGetTextureParameterIuiv", glGetTextureParameterIuiv == NULL);
  glewInfoFunc(fi, "glGetTextureParameterfv", glGetTextureParameterfv == NULL);
  glewInfoFunc(fi, "glGetTextureParameteriv", glGetTextureParameteriv == NULL);
  glewInfoFunc(fi, "glGetTransformFeedbacki64_v", glGetTransformFeedbacki64_v == NULL);
  glewInfoFunc(fi, "glGetTransformFeedbacki_v", glGetTransformFeedbacki_v == NULL);
  glewInfoFunc(fi, "glGetTransformFeedbackiv", glGetTransformFeedbackiv == NULL);
  glewInfoFunc(fi, "glGetVertexArrayIndexed64iv", glGetVertexArrayIndexed64iv == NULL);
  glewInfoFunc(fi, "glGetVertexArrayIndexediv", glGetVertexArrayIndexediv == NULL);
  glewInfoFunc(fi, "glGetVertexArrayiv", glGetVertexArrayiv == NULL);
  glewInfoFunc(fi, "glInvalidateNamedFramebufferData", glInvalidateNamedFramebufferData == NULL);
  glewInfoFunc(fi, "glInvalidateNamedFramebufferSubData", glInvalidateNamedFramebufferSubData == NULL);
  glewInfoFunc(fi, "glMapNamedBuffer", glMapNamedBuffer == NULL);
  glewInfoFunc(fi, "glMapNamedBufferRange", glMapNamedBufferRange == NULL);
  glewInfoFunc(fi, "glNamedBufferData", glNamedBufferData == NULL);
  glewInfoFunc(fi, "glNamedBufferStorage", glNamedBufferStorage == NULL);
  glewInfoFunc(fi, "glNamedBufferSubData", glNamedBufferSubData == NULL);
  glewInfoFunc(fi, "glNamedFramebufferDrawBuffer", glNamedFramebufferDrawBuffer == NULL);
  glewInfoFunc(fi, "glNamedFramebufferDrawBuffers", glNamedFramebufferDrawBuffers == NULL);
  glewInfoFunc(fi, "glNamedFramebufferParameteri", glNamedFramebufferParameteri == NULL);
  glewInfoFunc(fi, "glNamedFramebufferReadBuffer", glNamedFramebufferReadBuffer == NULL);
  glewInfoFunc(fi, "glNamedFramebufferRenderbuffer", glNamedFramebufferRenderbuffer == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTexture", glNamedFramebufferTexture == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTextureLayer", glNamedFramebufferTextureLayer == NULL);
  glewInfoFunc(fi, "glNamedRenderbufferStorage", glNamedRenderbufferStorage == NULL);
  glewInfoFunc(fi, "glNamedRenderbufferStorageMultisample", glNamedRenderbufferStorageMultisample == NULL);
  glewInfoFunc(fi, "glTextureBuffer", glTextureBuffer == NULL);
  glewInfoFunc(fi, "glTextureBufferRange", glTextureBufferRange == NULL);
  glewInfoFunc(fi, "glTextureParameterIiv", glTextureParameterIiv == NULL);
  glewInfoFunc(fi, "glTextureParameterIuiv", glTextureParameterIuiv == NULL);
  glewInfoFunc(fi, "glTextureParameterf", glTextureParameterf == NULL);
  glewInfoFunc(fi, "glTextureParameterfv", glTextureParameterfv == NULL);
  glewInfoFunc(fi, "glTextureParameteri", glTextureParameteri == NULL);
  glewInfoFunc(fi, "glTextureParameteriv", glTextureParameteriv == NULL);
  glewInfoFunc(fi, "glTextureStorage1D", glTextureStorage1D == NULL);
  glewInfoFunc(fi, "glTextureStorage2D", glTextureStorage2D == NULL);
  glewInfoFunc(fi, "glTextureStorage2DMultisample", glTextureStorage2DMultisample == NULL);
  glewInfoFunc(fi, "glTextureStorage3D", glTextureStorage3D == NULL);
  glewInfoFunc(fi, "glTextureStorage3DMultisample", glTextureStorage3DMultisample == NULL);
  glewInfoFunc(fi, "glTextureSubImage1D", glTextureSubImage1D == NULL);
  glewInfoFunc(fi, "glTextureSubImage2D", glTextureSubImage2D == NULL);
  glewInfoFunc(fi, "glTextureSubImage3D", glTextureSubImage3D == NULL);
  glewInfoFunc(fi, "glTransformFeedbackBufferBase", glTransformFeedbackBufferBase == NULL);
  glewInfoFunc(fi, "glTransformFeedbackBufferRange", glTransformFeedbackBufferRange == NULL);
  glewInfoFunc(fi, "glUnmapNamedBuffer", glUnmapNamedBuffer == NULL);
  glewInfoFunc(fi, "glVertexArrayAttribBinding", glVertexArrayAttribBinding == NULL);
  glewInfoFunc(fi, "glVertexArrayAttribFormat", glVertexArrayAttribFormat == NULL);
  glewInfoFunc(fi, "glVertexArrayAttribIFormat", glVertexArrayAttribIFormat == NULL);
  glewInfoFunc(fi, "glVertexArrayAttribLFormat", glVertexArrayAttribLFormat == NULL);
  glewInfoFunc(fi, "glVertexArrayBindingDivisor", glVertexArrayBindingDivisor == NULL);
  glewInfoFunc(fi, "glVertexArrayElementBuffer", glVertexArrayElementBuffer == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexBuffer", glVertexArrayVertexBuffer == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexBuffers", glVertexArrayVertexBuffers == NULL);
}

#endif /* GL_ARB_direct_state_access */

#ifdef GL_ARB_draw_buffers

static void _glewInfo_GL_ARB_draw_buffers (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_draw_buffers", GLEW_ARB_draw_buffers, glewIsSupported("GL_ARB_draw_buffers"), glewGetExtension("GL_ARB_draw_buffers"));

  glewInfoFunc(fi, "glDrawBuffersARB", glDrawBuffersARB == NULL);
}

#endif /* GL_ARB_draw_buffers */

#ifdef GL_ARB_draw_buffers_blend

static void _glewInfo_GL_ARB_draw_buffers_blend (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_draw_buffers_blend", GLEW_ARB_draw_buffers_blend, glewIsSupported("GL_ARB_draw_buffers_blend"), glewGetExtension("GL_ARB_draw_buffers_blend"));

  glewInfoFunc(fi, "glBlendEquationSeparateiARB", glBlendEquationSeparateiARB == NULL);
  glewInfoFunc(fi, "glBlendEquationiARB", glBlendEquationiARB == NULL);
  glewInfoFunc(fi, "glBlendFuncSeparateiARB", glBlendFuncSeparateiARB == NULL);
  glewInfoFunc(fi, "glBlendFunciARB", glBlendFunciARB == NULL);
}

#endif /* GL_ARB_draw_buffers_blend */

#ifdef GL_ARB_draw_elements_base_vertex

static void _glewInfo_GL_ARB_draw_elements_base_vertex (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_draw_elements_base_vertex", GLEW_ARB_draw_elements_base_vertex, glewIsSupported("GL_ARB_draw_elements_base_vertex"), glewGetExtension("GL_ARB_draw_elements_base_vertex"));

  glewInfoFunc(fi, "glDrawElementsBaseVertex", glDrawElementsBaseVertex == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedBaseVertex", glDrawElementsInstancedBaseVertex == NULL);
  glewInfoFunc(fi, "glDrawRangeElementsBaseVertex", glDrawRangeElementsBaseVertex == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsBaseVertex", glMultiDrawElementsBaseVertex == NULL);
}

#endif /* GL_ARB_draw_elements_base_vertex */

#ifdef GL_ARB_draw_indirect

static void _glewInfo_GL_ARB_draw_indirect (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_draw_indirect", GLEW_ARB_draw_indirect, glewIsSupported("GL_ARB_draw_indirect"), glewGetExtension("GL_ARB_draw_indirect"));

  glewInfoFunc(fi, "glDrawArraysIndirect", glDrawArraysIndirect == NULL);
  glewInfoFunc(fi, "glDrawElementsIndirect", glDrawElementsIndirect == NULL);
}

#endif /* GL_ARB_draw_indirect */

#ifdef GL_ARB_draw_instanced

static void _glewInfo_GL_ARB_draw_instanced (void)
{
  glewPrintExt("GL_ARB_draw_instanced", GLEW_ARB_draw_instanced, glewIsSupported("GL_ARB_draw_instanced"), glewGetExtension("GL_ARB_draw_instanced"));
}

#endif /* GL_ARB_draw_instanced */

#ifdef GL_ARB_enhanced_layouts

static void _glewInfo_GL_ARB_enhanced_layouts (void)
{
  glewPrintExt("GL_ARB_enhanced_layouts", GLEW_ARB_enhanced_layouts, glewIsSupported("GL_ARB_enhanced_layouts"), glewGetExtension("GL_ARB_enhanced_layouts"));
}

#endif /* GL_ARB_enhanced_layouts */

#ifdef GL_ARB_explicit_attrib_location

static void _glewInfo_GL_ARB_explicit_attrib_location (void)
{
  glewPrintExt("GL_ARB_explicit_attrib_location", GLEW_ARB_explicit_attrib_location, glewIsSupported("GL_ARB_explicit_attrib_location"), glewGetExtension("GL_ARB_explicit_attrib_location"));
}

#endif /* GL_ARB_explicit_attrib_location */

#ifdef GL_ARB_explicit_uniform_location

static void _glewInfo_GL_ARB_explicit_uniform_location (void)
{
  glewPrintExt("GL_ARB_explicit_uniform_location", GLEW_ARB_explicit_uniform_location, glewIsSupported("GL_ARB_explicit_uniform_location"), glewGetExtension("GL_ARB_explicit_uniform_location"));
}

#endif /* GL_ARB_explicit_uniform_location */

#ifdef GL_ARB_fragment_coord_conventions

static void _glewInfo_GL_ARB_fragment_coord_conventions (void)
{
  glewPrintExt("GL_ARB_fragment_coord_conventions", GLEW_ARB_fragment_coord_conventions, glewIsSupported("GL_ARB_fragment_coord_conventions"), glewGetExtension("GL_ARB_fragment_coord_conventions"));
}

#endif /* GL_ARB_fragment_coord_conventions */

#ifdef GL_ARB_fragment_layer_viewport

static void _glewInfo_GL_ARB_fragment_layer_viewport (void)
{
  glewPrintExt("GL_ARB_fragment_layer_viewport", GLEW_ARB_fragment_layer_viewport, glewIsSupported("GL_ARB_fragment_layer_viewport"), glewGetExtension("GL_ARB_fragment_layer_viewport"));
}

#endif /* GL_ARB_fragment_layer_viewport */

#ifdef GL_ARB_fragment_program

static void _glewInfo_GL_ARB_fragment_program (void)
{
  glewPrintExt("GL_ARB_fragment_program", GLEW_ARB_fragment_program, glewIsSupported("GL_ARB_fragment_program"), glewGetExtension("GL_ARB_fragment_program"));
}

#endif /* GL_ARB_fragment_program */

#ifdef GL_ARB_fragment_program_shadow

static void _glewInfo_GL_ARB_fragment_program_shadow (void)
{
  glewPrintExt("GL_ARB_fragment_program_shadow", GLEW_ARB_fragment_program_shadow, glewIsSupported("GL_ARB_fragment_program_shadow"), glewGetExtension("GL_ARB_fragment_program_shadow"));
}

#endif /* GL_ARB_fragment_program_shadow */

#ifdef GL_ARB_fragment_shader

static void _glewInfo_GL_ARB_fragment_shader (void)
{
  glewPrintExt("GL_ARB_fragment_shader", GLEW_ARB_fragment_shader, glewIsSupported("GL_ARB_fragment_shader"), glewGetExtension("GL_ARB_fragment_shader"));
}

#endif /* GL_ARB_fragment_shader */

#ifdef GL_ARB_fragment_shader_interlock

static void _glewInfo_GL_ARB_fragment_shader_interlock (void)
{
  glewPrintExt("GL_ARB_fragment_shader_interlock", GLEW_ARB_fragment_shader_interlock, glewIsSupported("GL_ARB_fragment_shader_interlock"), glewGetExtension("GL_ARB_fragment_shader_interlock"));
}

#endif /* GL_ARB_fragment_shader_interlock */

#ifdef GL_ARB_framebuffer_no_attachments

static void _glewInfo_GL_ARB_framebuffer_no_attachments (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_framebuffer_no_attachments", GLEW_ARB_framebuffer_no_attachments, glewIsSupported("GL_ARB_framebuffer_no_attachments"), glewGetExtension("GL_ARB_framebuffer_no_attachments"));

  glewInfoFunc(fi, "glFramebufferParameteri", glFramebufferParameteri == NULL);
  glewInfoFunc(fi, "glGetFramebufferParameteriv", glGetFramebufferParameteriv == NULL);
  glewInfoFunc(fi, "glGetNamedFramebufferParameterivEXT", glGetNamedFramebufferParameterivEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferParameteriEXT", glNamedFramebufferParameteriEXT == NULL);
}

#endif /* GL_ARB_framebuffer_no_attachments */

#ifdef GL_ARB_framebuffer_object

static void _glewInfo_GL_ARB_framebuffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_framebuffer_object", GLEW_ARB_framebuffer_object, glewIsSupported("GL_ARB_framebuffer_object"), glewGetExtension("GL_ARB_framebuffer_object"));

  glewInfoFunc(fi, "glBindFramebuffer", glBindFramebuffer == NULL);
  glewInfoFunc(fi, "glBindRenderbuffer", glBindRenderbuffer == NULL);
  glewInfoFunc(fi, "glBlitFramebuffer", glBlitFramebuffer == NULL);
  glewInfoFunc(fi, "glCheckFramebufferStatus", glCheckFramebufferStatus == NULL);
  glewInfoFunc(fi, "glDeleteFramebuffers", glDeleteFramebuffers == NULL);
  glewInfoFunc(fi, "glDeleteRenderbuffers", glDeleteRenderbuffers == NULL);
  glewInfoFunc(fi, "glFramebufferRenderbuffer", glFramebufferRenderbuffer == NULL);
  glewInfoFunc(fi, "glFramebufferTexture1D", glFramebufferTexture1D == NULL);
  glewInfoFunc(fi, "glFramebufferTexture2D", glFramebufferTexture2D == NULL);
  glewInfoFunc(fi, "glFramebufferTexture3D", glFramebufferTexture3D == NULL);
  glewInfoFunc(fi, "glFramebufferTextureLayer", glFramebufferTextureLayer == NULL);
  glewInfoFunc(fi, "glGenFramebuffers", glGenFramebuffers == NULL);
  glewInfoFunc(fi, "glGenRenderbuffers", glGenRenderbuffers == NULL);
  glewInfoFunc(fi, "glGenerateMipmap", glGenerateMipmap == NULL);
  glewInfoFunc(fi, "glGetFramebufferAttachmentParameteriv", glGetFramebufferAttachmentParameteriv == NULL);
  glewInfoFunc(fi, "glGetRenderbufferParameteriv", glGetRenderbufferParameteriv == NULL);
  glewInfoFunc(fi, "glIsFramebuffer", glIsFramebuffer == NULL);
  glewInfoFunc(fi, "glIsRenderbuffer", glIsRenderbuffer == NULL);
  glewInfoFunc(fi, "glRenderbufferStorage", glRenderbufferStorage == NULL);
  glewInfoFunc(fi, "glRenderbufferStorageMultisample", glRenderbufferStorageMultisample == NULL);
}

#endif /* GL_ARB_framebuffer_object */

#ifdef GL_ARB_framebuffer_sRGB

static void _glewInfo_GL_ARB_framebuffer_sRGB (void)
{
  glewPrintExt("GL_ARB_framebuffer_sRGB", GLEW_ARB_framebuffer_sRGB, glewIsSupported("GL_ARB_framebuffer_sRGB"), glewGetExtension("GL_ARB_framebuffer_sRGB"));
}

#endif /* GL_ARB_framebuffer_sRGB */

#ifdef GL_ARB_geometry_shader4

static void _glewInfo_GL_ARB_geometry_shader4 (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_geometry_shader4", GLEW_ARB_geometry_shader4, glewIsSupported("GL_ARB_geometry_shader4"), glewGetExtension("GL_ARB_geometry_shader4"));

  glewInfoFunc(fi, "glFramebufferTextureARB", glFramebufferTextureARB == NULL);
  glewInfoFunc(fi, "glFramebufferTextureFaceARB", glFramebufferTextureFaceARB == NULL);
  glewInfoFunc(fi, "glFramebufferTextureLayerARB", glFramebufferTextureLayerARB == NULL);
  glewInfoFunc(fi, "glProgramParameteriARB", glProgramParameteriARB == NULL);
}

#endif /* GL_ARB_geometry_shader4 */

#ifdef GL_ARB_get_program_binary

static void _glewInfo_GL_ARB_get_program_binary (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_get_program_binary", GLEW_ARB_get_program_binary, glewIsSupported("GL_ARB_get_program_binary"), glewGetExtension("GL_ARB_get_program_binary"));

  glewInfoFunc(fi, "glGetProgramBinary", glGetProgramBinary == NULL);
  glewInfoFunc(fi, "glProgramBinary", glProgramBinary == NULL);
  glewInfoFunc(fi, "glProgramParameteri", glProgramParameteri == NULL);
}

#endif /* GL_ARB_get_program_binary */

#ifdef GL_ARB_get_texture_sub_image

static void _glewInfo_GL_ARB_get_texture_sub_image (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_get_texture_sub_image", GLEW_ARB_get_texture_sub_image, glewIsSupported("GL_ARB_get_texture_sub_image"), glewGetExtension("GL_ARB_get_texture_sub_image"));

  glewInfoFunc(fi, "glGetCompressedTextureSubImage", glGetCompressedTextureSubImage == NULL);
  glewInfoFunc(fi, "glGetTextureSubImage", glGetTextureSubImage == NULL);
}

#endif /* GL_ARB_get_texture_sub_image */

#ifdef GL_ARB_gl_spirv

static void _glewInfo_GL_ARB_gl_spirv (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_gl_spirv", GLEW_ARB_gl_spirv, glewIsSupported("GL_ARB_gl_spirv"), glewGetExtension("GL_ARB_gl_spirv"));

  glewInfoFunc(fi, "glSpecializeShaderARB", glSpecializeShaderARB == NULL);
}

#endif /* GL_ARB_gl_spirv */

#ifdef GL_ARB_gpu_shader5

static void _glewInfo_GL_ARB_gpu_shader5 (void)
{
  glewPrintExt("GL_ARB_gpu_shader5", GLEW_ARB_gpu_shader5, glewIsSupported("GL_ARB_gpu_shader5"), glewGetExtension("GL_ARB_gpu_shader5"));
}

#endif /* GL_ARB_gpu_shader5 */

#ifdef GL_ARB_gpu_shader_fp64

static void _glewInfo_GL_ARB_gpu_shader_fp64 (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_gpu_shader_fp64", GLEW_ARB_gpu_shader_fp64, glewIsSupported("GL_ARB_gpu_shader_fp64"), glewGetExtension("GL_ARB_gpu_shader_fp64"));

  glewInfoFunc(fi, "glGetUniformdv", glGetUniformdv == NULL);
  glewInfoFunc(fi, "glUniform1d", glUniform1d == NULL);
  glewInfoFunc(fi, "glUniform1dv", glUniform1dv == NULL);
  glewInfoFunc(fi, "glUniform2d", glUniform2d == NULL);
  glewInfoFunc(fi, "glUniform2dv", glUniform2dv == NULL);
  glewInfoFunc(fi, "glUniform3d", glUniform3d == NULL);
  glewInfoFunc(fi, "glUniform3dv", glUniform3dv == NULL);
  glewInfoFunc(fi, "glUniform4d", glUniform4d == NULL);
  glewInfoFunc(fi, "glUniform4dv", glUniform4dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix2dv", glUniformMatrix2dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix2x3dv", glUniformMatrix2x3dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix2x4dv", glUniformMatrix2x4dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix3dv", glUniformMatrix3dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix3x2dv", glUniformMatrix3x2dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix3x4dv", glUniformMatrix3x4dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix4dv", glUniformMatrix4dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix4x2dv", glUniformMatrix4x2dv == NULL);
  glewInfoFunc(fi, "glUniformMatrix4x3dv", glUniformMatrix4x3dv == NULL);
}

#endif /* GL_ARB_gpu_shader_fp64 */

#ifdef GL_ARB_gpu_shader_int64

static void _glewInfo_GL_ARB_gpu_shader_int64 (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_gpu_shader_int64", GLEW_ARB_gpu_shader_int64, glewIsSupported("GL_ARB_gpu_shader_int64"), glewGetExtension("GL_ARB_gpu_shader_int64"));

  glewInfoFunc(fi, "glGetUniformi64vARB", glGetUniformi64vARB == NULL);
  glewInfoFunc(fi, "glGetUniformui64vARB", glGetUniformui64vARB == NULL);
  glewInfoFunc(fi, "glGetnUniformi64vARB", glGetnUniformi64vARB == NULL);
  glewInfoFunc(fi, "glGetnUniformui64vARB", glGetnUniformui64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform1i64ARB", glProgramUniform1i64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform1i64vARB", glProgramUniform1i64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform1ui64ARB", glProgramUniform1ui64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform1ui64vARB", glProgramUniform1ui64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform2i64ARB", glProgramUniform2i64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform2i64vARB", glProgramUniform2i64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform2ui64ARB", glProgramUniform2ui64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform2ui64vARB", glProgramUniform2ui64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform3i64ARB", glProgramUniform3i64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform3i64vARB", glProgramUniform3i64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform3ui64ARB", glProgramUniform3ui64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform3ui64vARB", glProgramUniform3ui64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform4i64ARB", glProgramUniform4i64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform4i64vARB", glProgramUniform4i64vARB == NULL);
  glewInfoFunc(fi, "glProgramUniform4ui64ARB", glProgramUniform4ui64ARB == NULL);
  glewInfoFunc(fi, "glProgramUniform4ui64vARB", glProgramUniform4ui64vARB == NULL);
  glewInfoFunc(fi, "glUniform1i64ARB", glUniform1i64ARB == NULL);
  glewInfoFunc(fi, "glUniform1i64vARB", glUniform1i64vARB == NULL);
  glewInfoFunc(fi, "glUniform1ui64ARB", glUniform1ui64ARB == NULL);
  glewInfoFunc(fi, "glUniform1ui64vARB", glUniform1ui64vARB == NULL);
  glewInfoFunc(fi, "glUniform2i64ARB", glUniform2i64ARB == NULL);
  glewInfoFunc(fi, "glUniform2i64vARB", glUniform2i64vARB == NULL);
  glewInfoFunc(fi, "glUniform2ui64ARB", glUniform2ui64ARB == NULL);
  glewInfoFunc(fi, "glUniform2ui64vARB", glUniform2ui64vARB == NULL);
  glewInfoFunc(fi, "glUniform3i64ARB", glUniform3i64ARB == NULL);
  glewInfoFunc(fi, "glUniform3i64vARB", glUniform3i64vARB == NULL);
  glewInfoFunc(fi, "glUniform3ui64ARB", glUniform3ui64ARB == NULL);
  glewInfoFunc(fi, "glUniform3ui64vARB", glUniform3ui64vARB == NULL);
  glewInfoFunc(fi, "glUniform4i64ARB", glUniform4i64ARB == NULL);
  glewInfoFunc(fi, "glUniform4i64vARB", glUniform4i64vARB == NULL);
  glewInfoFunc(fi, "glUniform4ui64ARB", glUniform4ui64ARB == NULL);
  glewInfoFunc(fi, "glUniform4ui64vARB", glUniform4ui64vARB == NULL);
}

#endif /* GL_ARB_gpu_shader_int64 */

#ifdef GL_ARB_half_float_pixel

static void _glewInfo_GL_ARB_half_float_pixel (void)
{
  glewPrintExt("GL_ARB_half_float_pixel", GLEW_ARB_half_float_pixel, glewIsSupported("GL_ARB_half_float_pixel"), glewGetExtension("GL_ARB_half_float_pixel"));
}

#endif /* GL_ARB_half_float_pixel */

#ifdef GL_ARB_half_float_vertex

static void _glewInfo_GL_ARB_half_float_vertex (void)
{
  glewPrintExt("GL_ARB_half_float_vertex", GLEW_ARB_half_float_vertex, glewIsSupported("GL_ARB_half_float_vertex"), glewGetExtension("GL_ARB_half_float_vertex"));
}

#endif /* GL_ARB_half_float_vertex */

#ifdef GL_ARB_imaging

static void _glewInfo_GL_ARB_imaging (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_imaging", GLEW_ARB_imaging, glewIsSupported("GL_ARB_imaging"), glewGetExtension("GL_ARB_imaging"));

  glewInfoFunc(fi, "glBlendEquation", glBlendEquation == NULL);
  glewInfoFunc(fi, "glColorSubTable", glColorSubTable == NULL);
  glewInfoFunc(fi, "glColorTable", glColorTable == NULL);
  glewInfoFunc(fi, "glColorTableParameterfv", glColorTableParameterfv == NULL);
  glewInfoFunc(fi, "glColorTableParameteriv", glColorTableParameteriv == NULL);
  glewInfoFunc(fi, "glConvolutionFilter1D", glConvolutionFilter1D == NULL);
  glewInfoFunc(fi, "glConvolutionFilter2D", glConvolutionFilter2D == NULL);
  glewInfoFunc(fi, "glConvolutionParameterf", glConvolutionParameterf == NULL);
  glewInfoFunc(fi, "glConvolutionParameterfv", glConvolutionParameterfv == NULL);
  glewInfoFunc(fi, "glConvolutionParameteri", glConvolutionParameteri == NULL);
  glewInfoFunc(fi, "glConvolutionParameteriv", glConvolutionParameteriv == NULL);
  glewInfoFunc(fi, "glCopyColorSubTable", glCopyColorSubTable == NULL);
  glewInfoFunc(fi, "glCopyColorTable", glCopyColorTable == NULL);
  glewInfoFunc(fi, "glCopyConvolutionFilter1D", glCopyConvolutionFilter1D == NULL);
  glewInfoFunc(fi, "glCopyConvolutionFilter2D", glCopyConvolutionFilter2D == NULL);
  glewInfoFunc(fi, "glGetColorTable", glGetColorTable == NULL);
  glewInfoFunc(fi, "glGetColorTableParameterfv", glGetColorTableParameterfv == NULL);
  glewInfoFunc(fi, "glGetColorTableParameteriv", glGetColorTableParameteriv == NULL);
  glewInfoFunc(fi, "glGetConvolutionFilter", glGetConvolutionFilter == NULL);
  glewInfoFunc(fi, "glGetConvolutionParameterfv", glGetConvolutionParameterfv == NULL);
  glewInfoFunc(fi, "glGetConvolutionParameteriv", glGetConvolutionParameteriv == NULL);
  glewInfoFunc(fi, "glGetHistogram", glGetHistogram == NULL);
  glewInfoFunc(fi, "glGetHistogramParameterfv", glGetHistogramParameterfv == NULL);
  glewInfoFunc(fi, "glGetHistogramParameteriv", glGetHistogramParameteriv == NULL);
  glewInfoFunc(fi, "glGetMinmax", glGetMinmax == NULL);
  glewInfoFunc(fi, "glGetMinmaxParameterfv", glGetMinmaxParameterfv == NULL);
  glewInfoFunc(fi, "glGetMinmaxParameteriv", glGetMinmaxParameteriv == NULL);
  glewInfoFunc(fi, "glGetSeparableFilter", glGetSeparableFilter == NULL);
  glewInfoFunc(fi, "glHistogram", glHistogram == NULL);
  glewInfoFunc(fi, "glMinmax", glMinmax == NULL);
  glewInfoFunc(fi, "glResetHistogram", glResetHistogram == NULL);
  glewInfoFunc(fi, "glResetMinmax", glResetMinmax == NULL);
  glewInfoFunc(fi, "glSeparableFilter2D", glSeparableFilter2D == NULL);
}

#endif /* GL_ARB_imaging */

#ifdef GL_ARB_indirect_parameters

static void _glewInfo_GL_ARB_indirect_parameters (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_indirect_parameters", GLEW_ARB_indirect_parameters, glewIsSupported("GL_ARB_indirect_parameters"), glewGetExtension("GL_ARB_indirect_parameters"));

  glewInfoFunc(fi, "glMultiDrawArraysIndirectCountARB", glMultiDrawArraysIndirectCountARB == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsIndirectCountARB", glMultiDrawElementsIndirectCountARB == NULL);
}

#endif /* GL_ARB_indirect_parameters */

#ifdef GL_ARB_instanced_arrays

static void _glewInfo_GL_ARB_instanced_arrays (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_instanced_arrays", GLEW_ARB_instanced_arrays, glewIsSupported("GL_ARB_instanced_arrays"), glewGetExtension("GL_ARB_instanced_arrays"));

  glewInfoFunc(fi, "glDrawArraysInstancedARB", glDrawArraysInstancedARB == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedARB", glDrawElementsInstancedARB == NULL);
  glewInfoFunc(fi, "glVertexAttribDivisorARB", glVertexAttribDivisorARB == NULL);
}

#endif /* GL_ARB_instanced_arrays */

#ifdef GL_ARB_internalformat_query

static void _glewInfo_GL_ARB_internalformat_query (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_internalformat_query", GLEW_ARB_internalformat_query, glewIsSupported("GL_ARB_internalformat_query"), glewGetExtension("GL_ARB_internalformat_query"));

  glewInfoFunc(fi, "glGetInternalformativ", glGetInternalformativ == NULL);
}

#endif /* GL_ARB_internalformat_query */

#ifdef GL_ARB_internalformat_query2

static void _glewInfo_GL_ARB_internalformat_query2 (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_internalformat_query2", GLEW_ARB_internalformat_query2, glewIsSupported("GL_ARB_internalformat_query2"), glewGetExtension("GL_ARB_internalformat_query2"));

  glewInfoFunc(fi, "glGetInternalformati64v", glGetInternalformati64v == NULL);
}

#endif /* GL_ARB_internalformat_query2 */

#ifdef GL_ARB_invalidate_subdata

static void _glewInfo_GL_ARB_invalidate_subdata (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_invalidate_subdata", GLEW_ARB_invalidate_subdata, glewIsSupported("GL_ARB_invalidate_subdata"), glewGetExtension("GL_ARB_invalidate_subdata"));

  glewInfoFunc(fi, "glInvalidateBufferData", glInvalidateBufferData == NULL);
  glewInfoFunc(fi, "glInvalidateBufferSubData", glInvalidateBufferSubData == NULL);
  glewInfoFunc(fi, "glInvalidateFramebuffer", glInvalidateFramebuffer == NULL);
  glewInfoFunc(fi, "glInvalidateSubFramebuffer", glInvalidateSubFramebuffer == NULL);
  glewInfoFunc(fi, "glInvalidateTexImage", glInvalidateTexImage == NULL);
  glewInfoFunc(fi, "glInvalidateTexSubImage", glInvalidateTexSubImage == NULL);
}

#endif /* GL_ARB_invalidate_subdata */

#ifdef GL_ARB_map_buffer_alignment

static void _glewInfo_GL_ARB_map_buffer_alignment (void)
{
  glewPrintExt("GL_ARB_map_buffer_alignment", GLEW_ARB_map_buffer_alignment, glewIsSupported("GL_ARB_map_buffer_alignment"), glewGetExtension("GL_ARB_map_buffer_alignment"));
}

#endif /* GL_ARB_map_buffer_alignment */

#ifdef GL_ARB_map_buffer_range

static void _glewInfo_GL_ARB_map_buffer_range (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_map_buffer_range", GLEW_ARB_map_buffer_range, glewIsSupported("GL_ARB_map_buffer_range"), glewGetExtension("GL_ARB_map_buffer_range"));

  glewInfoFunc(fi, "glFlushMappedBufferRange", glFlushMappedBufferRange == NULL);
  glewInfoFunc(fi, "glMapBufferRange", glMapBufferRange == NULL);
}

#endif /* GL_ARB_map_buffer_range */

#ifdef GL_ARB_matrix_palette

static void _glewInfo_GL_ARB_matrix_palette (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_matrix_palette", GLEW_ARB_matrix_palette, glewIsSupported("GL_ARB_matrix_palette"), glewGetExtension("GL_ARB_matrix_palette"));

  glewInfoFunc(fi, "glCurrentPaletteMatrixARB", glCurrentPaletteMatrixARB == NULL);
  glewInfoFunc(fi, "glMatrixIndexPointerARB", glMatrixIndexPointerARB == NULL);
  glewInfoFunc(fi, "glMatrixIndexubvARB", glMatrixIndexubvARB == NULL);
  glewInfoFunc(fi, "glMatrixIndexuivARB", glMatrixIndexuivARB == NULL);
  glewInfoFunc(fi, "glMatrixIndexusvARB", glMatrixIndexusvARB == NULL);
}

#endif /* GL_ARB_matrix_palette */

#ifdef GL_ARB_multi_bind

static void _glewInfo_GL_ARB_multi_bind (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_multi_bind", GLEW_ARB_multi_bind, glewIsSupported("GL_ARB_multi_bind"), glewGetExtension("GL_ARB_multi_bind"));

  glewInfoFunc(fi, "glBindBuffersBase", glBindBuffersBase == NULL);
  glewInfoFunc(fi, "glBindBuffersRange", glBindBuffersRange == NULL);
  glewInfoFunc(fi, "glBindImageTextures", glBindImageTextures == NULL);
  glewInfoFunc(fi, "glBindSamplers", glBindSamplers == NULL);
  glewInfoFunc(fi, "glBindTextures", glBindTextures == NULL);
  glewInfoFunc(fi, "glBindVertexBuffers", glBindVertexBuffers == NULL);
}

#endif /* GL_ARB_multi_bind */

#ifdef GL_ARB_multi_draw_indirect

static void _glewInfo_GL_ARB_multi_draw_indirect (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_multi_draw_indirect", GLEW_ARB_multi_draw_indirect, glewIsSupported("GL_ARB_multi_draw_indirect"), glewGetExtension("GL_ARB_multi_draw_indirect"));

  glewInfoFunc(fi, "glMultiDrawArraysIndirect", glMultiDrawArraysIndirect == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsIndirect", glMultiDrawElementsIndirect == NULL);
}

#endif /* GL_ARB_multi_draw_indirect */

#ifdef GL_ARB_multisample

static void _glewInfo_GL_ARB_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_multisample", GLEW_ARB_multisample, glewIsSupported("GL_ARB_multisample"), glewGetExtension("GL_ARB_multisample"));

  glewInfoFunc(fi, "glSampleCoverageARB", glSampleCoverageARB == NULL);
}

#endif /* GL_ARB_multisample */

#ifdef GL_ARB_multitexture

static void _glewInfo_GL_ARB_multitexture (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_multitexture", GLEW_ARB_multitexture, glewIsSupported("GL_ARB_multitexture"), glewGetExtension("GL_ARB_multitexture"));

  glewInfoFunc(fi, "glActiveTextureARB", glActiveTextureARB == NULL);
  glewInfoFunc(fi, "glClientActiveTextureARB", glClientActiveTextureARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1dARB", glMultiTexCoord1dARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1dvARB", glMultiTexCoord1dvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1fARB", glMultiTexCoord1fARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1fvARB", glMultiTexCoord1fvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1iARB", glMultiTexCoord1iARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1ivARB", glMultiTexCoord1ivARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1sARB", glMultiTexCoord1sARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1svARB", glMultiTexCoord1svARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2dARB", glMultiTexCoord2dARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2dvARB", glMultiTexCoord2dvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2fARB", glMultiTexCoord2fARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2fvARB", glMultiTexCoord2fvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2iARB", glMultiTexCoord2iARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2ivARB", glMultiTexCoord2ivARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2sARB", glMultiTexCoord2sARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2svARB", glMultiTexCoord2svARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3dARB", glMultiTexCoord3dARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3dvARB", glMultiTexCoord3dvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3fARB", glMultiTexCoord3fARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3fvARB", glMultiTexCoord3fvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3iARB", glMultiTexCoord3iARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3ivARB", glMultiTexCoord3ivARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3sARB", glMultiTexCoord3sARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3svARB", glMultiTexCoord3svARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4dARB", glMultiTexCoord4dARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4dvARB", glMultiTexCoord4dvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4fARB", glMultiTexCoord4fARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4fvARB", glMultiTexCoord4fvARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4iARB", glMultiTexCoord4iARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4ivARB", glMultiTexCoord4ivARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4sARB", glMultiTexCoord4sARB == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4svARB", glMultiTexCoord4svARB == NULL);
}

#endif /* GL_ARB_multitexture */

#ifdef GL_ARB_occlusion_query

static void _glewInfo_GL_ARB_occlusion_query (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_occlusion_query", GLEW_ARB_occlusion_query, glewIsSupported("GL_ARB_occlusion_query"), glewGetExtension("GL_ARB_occlusion_query"));

  glewInfoFunc(fi, "glBeginQueryARB", glBeginQueryARB == NULL);
  glewInfoFunc(fi, "glDeleteQueriesARB", glDeleteQueriesARB == NULL);
  glewInfoFunc(fi, "glEndQueryARB", glEndQueryARB == NULL);
  glewInfoFunc(fi, "glGenQueriesARB", glGenQueriesARB == NULL);
  glewInfoFunc(fi, "glGetQueryObjectivARB", glGetQueryObjectivARB == NULL);
  glewInfoFunc(fi, "glGetQueryObjectuivARB", glGetQueryObjectuivARB == NULL);
  glewInfoFunc(fi, "glGetQueryivARB", glGetQueryivARB == NULL);
  glewInfoFunc(fi, "glIsQueryARB", glIsQueryARB == NULL);
}

#endif /* GL_ARB_occlusion_query */

#ifdef GL_ARB_occlusion_query2

static void _glewInfo_GL_ARB_occlusion_query2 (void)
{
  glewPrintExt("GL_ARB_occlusion_query2", GLEW_ARB_occlusion_query2, glewIsSupported("GL_ARB_occlusion_query2"), glewGetExtension("GL_ARB_occlusion_query2"));
}

#endif /* GL_ARB_occlusion_query2 */

#ifdef GL_ARB_parallel_shader_compile

static void _glewInfo_GL_ARB_parallel_shader_compile (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_parallel_shader_compile", GLEW_ARB_parallel_shader_compile, glewIsSupported("GL_ARB_parallel_shader_compile"), glewGetExtension("GL_ARB_parallel_shader_compile"));

  glewInfoFunc(fi, "glMaxShaderCompilerThreadsARB", glMaxShaderCompilerThreadsARB == NULL);
}

#endif /* GL_ARB_parallel_shader_compile */

#ifdef GL_ARB_pipeline_statistics_query

static void _glewInfo_GL_ARB_pipeline_statistics_query (void)
{
  glewPrintExt("GL_ARB_pipeline_statistics_query", GLEW_ARB_pipeline_statistics_query, glewIsSupported("GL_ARB_pipeline_statistics_query"), glewGetExtension("GL_ARB_pipeline_statistics_query"));
}

#endif /* GL_ARB_pipeline_statistics_query */

#ifdef GL_ARB_pixel_buffer_object

static void _glewInfo_GL_ARB_pixel_buffer_object (void)
{
  glewPrintExt("GL_ARB_pixel_buffer_object", GLEW_ARB_pixel_buffer_object, glewIsSupported("GL_ARB_pixel_buffer_object"), glewGetExtension("GL_ARB_pixel_buffer_object"));
}

#endif /* GL_ARB_pixel_buffer_object */

#ifdef GL_ARB_point_parameters

static void _glewInfo_GL_ARB_point_parameters (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_point_parameters", GLEW_ARB_point_parameters, glewIsSupported("GL_ARB_point_parameters"), glewGetExtension("GL_ARB_point_parameters"));

  glewInfoFunc(fi, "glPointParameterfARB", glPointParameterfARB == NULL);
  glewInfoFunc(fi, "glPointParameterfvARB", glPointParameterfvARB == NULL);
}

#endif /* GL_ARB_point_parameters */

#ifdef GL_ARB_point_sprite

static void _glewInfo_GL_ARB_point_sprite (void)
{
  glewPrintExt("GL_ARB_point_sprite", GLEW_ARB_point_sprite, glewIsSupported("GL_ARB_point_sprite"), glewGetExtension("GL_ARB_point_sprite"));
}

#endif /* GL_ARB_point_sprite */

#ifdef GL_ARB_polygon_offset_clamp

static void _glewInfo_GL_ARB_polygon_offset_clamp (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_polygon_offset_clamp", GLEW_ARB_polygon_offset_clamp, glewIsSupported("GL_ARB_polygon_offset_clamp"), glewGetExtension("GL_ARB_polygon_offset_clamp"));

  glewInfoFunc(fi, "glPolygonOffsetClamp", glPolygonOffsetClamp == NULL);
}

#endif /* GL_ARB_polygon_offset_clamp */

#ifdef GL_ARB_post_depth_coverage

static void _glewInfo_GL_ARB_post_depth_coverage (void)
{
  glewPrintExt("GL_ARB_post_depth_coverage", GLEW_ARB_post_depth_coverage, glewIsSupported("GL_ARB_post_depth_coverage"), glewGetExtension("GL_ARB_post_depth_coverage"));
}

#endif /* GL_ARB_post_depth_coverage */

#ifdef GL_ARB_program_interface_query

static void _glewInfo_GL_ARB_program_interface_query (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_program_interface_query", GLEW_ARB_program_interface_query, glewIsSupported("GL_ARB_program_interface_query"), glewGetExtension("GL_ARB_program_interface_query"));

  glewInfoFunc(fi, "glGetProgramInterfaceiv", glGetProgramInterfaceiv == NULL);
  glewInfoFunc(fi, "glGetProgramResourceIndex", glGetProgramResourceIndex == NULL);
  glewInfoFunc(fi, "glGetProgramResourceLocation", glGetProgramResourceLocation == NULL);
  glewInfoFunc(fi, "glGetProgramResourceLocationIndex", glGetProgramResourceLocationIndex == NULL);
  glewInfoFunc(fi, "glGetProgramResourceName", glGetProgramResourceName == NULL);
  glewInfoFunc(fi, "glGetProgramResourceiv", glGetProgramResourceiv == NULL);
}

#endif /* GL_ARB_program_interface_query */

#ifdef GL_ARB_provoking_vertex

static void _glewInfo_GL_ARB_provoking_vertex (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_provoking_vertex", GLEW_ARB_provoking_vertex, glewIsSupported("GL_ARB_provoking_vertex"), glewGetExtension("GL_ARB_provoking_vertex"));

  glewInfoFunc(fi, "glProvokingVertex", glProvokingVertex == NULL);
}

#endif /* GL_ARB_provoking_vertex */

#ifdef GL_ARB_query_buffer_object

static void _glewInfo_GL_ARB_query_buffer_object (void)
{
  glewPrintExt("GL_ARB_query_buffer_object", GLEW_ARB_query_buffer_object, glewIsSupported("GL_ARB_query_buffer_object"), glewGetExtension("GL_ARB_query_buffer_object"));
}

#endif /* GL_ARB_query_buffer_object */

#ifdef GL_ARB_robust_buffer_access_behavior

static void _glewInfo_GL_ARB_robust_buffer_access_behavior (void)
{
  glewPrintExt("GL_ARB_robust_buffer_access_behavior", GLEW_ARB_robust_buffer_access_behavior, glewIsSupported("GL_ARB_robust_buffer_access_behavior"), glewGetExtension("GL_ARB_robust_buffer_access_behavior"));
}

#endif /* GL_ARB_robust_buffer_access_behavior */

#ifdef GL_ARB_robustness

static void _glewInfo_GL_ARB_robustness (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_robustness", GLEW_ARB_robustness, glewIsSupported("GL_ARB_robustness"), glewGetExtension("GL_ARB_robustness"));

  glewInfoFunc(fi, "glGetGraphicsResetStatusARB", glGetGraphicsResetStatusARB == NULL);
  glewInfoFunc(fi, "glGetnColorTableARB", glGetnColorTableARB == NULL);
  glewInfoFunc(fi, "glGetnCompressedTexImageARB", glGetnCompressedTexImageARB == NULL);
  glewInfoFunc(fi, "glGetnConvolutionFilterARB", glGetnConvolutionFilterARB == NULL);
  glewInfoFunc(fi, "glGetnHistogramARB", glGetnHistogramARB == NULL);
  glewInfoFunc(fi, "glGetnMapdvARB", glGetnMapdvARB == NULL);
  glewInfoFunc(fi, "glGetnMapfvARB", glGetnMapfvARB == NULL);
  glewInfoFunc(fi, "glGetnMapivARB", glGetnMapivARB == NULL);
  glewInfoFunc(fi, "glGetnMinmaxARB", glGetnMinmaxARB == NULL);
  glewInfoFunc(fi, "glGetnPixelMapfvARB", glGetnPixelMapfvARB == NULL);
  glewInfoFunc(fi, "glGetnPixelMapuivARB", glGetnPixelMapuivARB == NULL);
  glewInfoFunc(fi, "glGetnPixelMapusvARB", glGetnPixelMapusvARB == NULL);
  glewInfoFunc(fi, "glGetnPolygonStippleARB", glGetnPolygonStippleARB == NULL);
  glewInfoFunc(fi, "glGetnSeparableFilterARB", glGetnSeparableFilterARB == NULL);
  glewInfoFunc(fi, "glGetnTexImageARB", glGetnTexImageARB == NULL);
  glewInfoFunc(fi, "glGetnUniformdvARB", glGetnUniformdvARB == NULL);
  glewInfoFunc(fi, "glGetnUniformfvARB", glGetnUniformfvARB == NULL);
  glewInfoFunc(fi, "glGetnUniformivARB", glGetnUniformivARB == NULL);
  glewInfoFunc(fi, "glGetnUniformuivARB", glGetnUniformuivARB == NULL);
  glewInfoFunc(fi, "glReadnPixelsARB", glReadnPixelsARB == NULL);
}

#endif /* GL_ARB_robustness */

#ifdef GL_ARB_robustness_application_isolation

static void _glewInfo_GL_ARB_robustness_application_isolation (void)
{
  glewPrintExt("GL_ARB_robustness_application_isolation", GLEW_ARB_robustness_application_isolation, glewIsSupported("GL_ARB_robustness_application_isolation"), glewGetExtension("GL_ARB_robustness_application_isolation"));
}

#endif /* GL_ARB_robustness_application_isolation */

#ifdef GL_ARB_robustness_share_group_isolation

static void _glewInfo_GL_ARB_robustness_share_group_isolation (void)
{
  glewPrintExt("GL_ARB_robustness_share_group_isolation", GLEW_ARB_robustness_share_group_isolation, glewIsSupported("GL_ARB_robustness_share_group_isolation"), glewGetExtension("GL_ARB_robustness_share_group_isolation"));
}

#endif /* GL_ARB_robustness_share_group_isolation */

#ifdef GL_ARB_sample_locations

static void _glewInfo_GL_ARB_sample_locations (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_sample_locations", GLEW_ARB_sample_locations, glewIsSupported("GL_ARB_sample_locations"), glewGetExtension("GL_ARB_sample_locations"));

  glewInfoFunc(fi, "glFramebufferSampleLocationsfvARB", glFramebufferSampleLocationsfvARB == NULL);
  glewInfoFunc(fi, "glNamedFramebufferSampleLocationsfvARB", glNamedFramebufferSampleLocationsfvARB == NULL);
}

#endif /* GL_ARB_sample_locations */

#ifdef GL_ARB_sample_shading

static void _glewInfo_GL_ARB_sample_shading (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_sample_shading", GLEW_ARB_sample_shading, glewIsSupported("GL_ARB_sample_shading"), glewGetExtension("GL_ARB_sample_shading"));

  glewInfoFunc(fi, "glMinSampleShadingARB", glMinSampleShadingARB == NULL);
}

#endif /* GL_ARB_sample_shading */

#ifdef GL_ARB_sampler_objects

static void _glewInfo_GL_ARB_sampler_objects (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_sampler_objects", GLEW_ARB_sampler_objects, glewIsSupported("GL_ARB_sampler_objects"), glewGetExtension("GL_ARB_sampler_objects"));

  glewInfoFunc(fi, "glBindSampler", glBindSampler == NULL);
  glewInfoFunc(fi, "glDeleteSamplers", glDeleteSamplers == NULL);
  glewInfoFunc(fi, "glGenSamplers", glGenSamplers == NULL);
  glewInfoFunc(fi, "glGetSamplerParameterIiv", glGetSamplerParameterIiv == NULL);
  glewInfoFunc(fi, "glGetSamplerParameterIuiv", glGetSamplerParameterIuiv == NULL);
  glewInfoFunc(fi, "glGetSamplerParameterfv", glGetSamplerParameterfv == NULL);
  glewInfoFunc(fi, "glGetSamplerParameteriv", glGetSamplerParameteriv == NULL);
  glewInfoFunc(fi, "glIsSampler", glIsSampler == NULL);
  glewInfoFunc(fi, "glSamplerParameterIiv", glSamplerParameterIiv == NULL);
  glewInfoFunc(fi, "glSamplerParameterIuiv", glSamplerParameterIuiv == NULL);
  glewInfoFunc(fi, "glSamplerParameterf", glSamplerParameterf == NULL);
  glewInfoFunc(fi, "glSamplerParameterfv", glSamplerParameterfv == NULL);
  glewInfoFunc(fi, "glSamplerParameteri", glSamplerParameteri == NULL);
  glewInfoFunc(fi, "glSamplerParameteriv", glSamplerParameteriv == NULL);
}

#endif /* GL_ARB_sampler_objects */

#ifdef GL_ARB_seamless_cube_map

static void _glewInfo_GL_ARB_seamless_cube_map (void)
{
  glewPrintExt("GL_ARB_seamless_cube_map", GLEW_ARB_seamless_cube_map, glewIsSupported("GL_ARB_seamless_cube_map"), glewGetExtension("GL_ARB_seamless_cube_map"));
}

#endif /* GL_ARB_seamless_cube_map */

#ifdef GL_ARB_seamless_cubemap_per_texture

static void _glewInfo_GL_ARB_seamless_cubemap_per_texture (void)
{
  glewPrintExt("GL_ARB_seamless_cubemap_per_texture", GLEW_ARB_seamless_cubemap_per_texture, glewIsSupported("GL_ARB_seamless_cubemap_per_texture"), glewGetExtension("GL_ARB_seamless_cubemap_per_texture"));
}

#endif /* GL_ARB_seamless_cubemap_per_texture */

#ifdef GL_ARB_separate_shader_objects

static void _glewInfo_GL_ARB_separate_shader_objects (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_separate_shader_objects", GLEW_ARB_separate_shader_objects, glewIsSupported("GL_ARB_separate_shader_objects"), glewGetExtension("GL_ARB_separate_shader_objects"));

  glewInfoFunc(fi, "glActiveShaderProgram", glActiveShaderProgram == NULL);
  glewInfoFunc(fi, "glBindProgramPipeline", glBindProgramPipeline == NULL);
  glewInfoFunc(fi, "glCreateShaderProgramv", glCreateShaderProgramv == NULL);
  glewInfoFunc(fi, "glDeleteProgramPipelines", glDeleteProgramPipelines == NULL);
  glewInfoFunc(fi, "glGenProgramPipelines", glGenProgramPipelines == NULL);
  glewInfoFunc(fi, "glGetProgramPipelineInfoLog", glGetProgramPipelineInfoLog == NULL);
  glewInfoFunc(fi, "glGetProgramPipelineiv", glGetProgramPipelineiv == NULL);
  glewInfoFunc(fi, "glIsProgramPipeline", glIsProgramPipeline == NULL);
  glewInfoFunc(fi, "glProgramUniform1d", glProgramUniform1d == NULL);
  glewInfoFunc(fi, "glProgramUniform1dv", glProgramUniform1dv == NULL);
  glewInfoFunc(fi, "glProgramUniform1f", glProgramUniform1f == NULL);
  glewInfoFunc(fi, "glProgramUniform1fv", glProgramUniform1fv == NULL);
  glewInfoFunc(fi, "glProgramUniform1i", glProgramUniform1i == NULL);
  glewInfoFunc(fi, "glProgramUniform1iv", glProgramUniform1iv == NULL);
  glewInfoFunc(fi, "glProgramUniform1ui", glProgramUniform1ui == NULL);
  glewInfoFunc(fi, "glProgramUniform1uiv", glProgramUniform1uiv == NULL);
  glewInfoFunc(fi, "glProgramUniform2d", glProgramUniform2d == NULL);
  glewInfoFunc(fi, "glProgramUniform2dv", glProgramUniform2dv == NULL);
  glewInfoFunc(fi, "glProgramUniform2f", glProgramUniform2f == NULL);
  glewInfoFunc(fi, "glProgramUniform2fv", glProgramUniform2fv == NULL);
  glewInfoFunc(fi, "glProgramUniform2i", glProgramUniform2i == NULL);
  glewInfoFunc(fi, "glProgramUniform2iv", glProgramUniform2iv == NULL);
  glewInfoFunc(fi, "glProgramUniform2ui", glProgramUniform2ui == NULL);
  glewInfoFunc(fi, "glProgramUniform2uiv", glProgramUniform2uiv == NULL);
  glewInfoFunc(fi, "glProgramUniform3d", glProgramUniform3d == NULL);
  glewInfoFunc(fi, "glProgramUniform3dv", glProgramUniform3dv == NULL);
  glewInfoFunc(fi, "glProgramUniform3f", glProgramUniform3f == NULL);
  glewInfoFunc(fi, "glProgramUniform3fv", glProgramUniform3fv == NULL);
  glewInfoFunc(fi, "glProgramUniform3i", glProgramUniform3i == NULL);
  glewInfoFunc(fi, "glProgramUniform3iv", glProgramUniform3iv == NULL);
  glewInfoFunc(fi, "glProgramUniform3ui", glProgramUniform3ui == NULL);
  glewInfoFunc(fi, "glProgramUniform3uiv", glProgramUniform3uiv == NULL);
  glewInfoFunc(fi, "glProgramUniform4d", glProgramUniform4d == NULL);
  glewInfoFunc(fi, "glProgramUniform4dv", glProgramUniform4dv == NULL);
  glewInfoFunc(fi, "glProgramUniform4f", glProgramUniform4f == NULL);
  glewInfoFunc(fi, "glProgramUniform4fv", glProgramUniform4fv == NULL);
  glewInfoFunc(fi, "glProgramUniform4i", glProgramUniform4i == NULL);
  glewInfoFunc(fi, "glProgramUniform4iv", glProgramUniform4iv == NULL);
  glewInfoFunc(fi, "glProgramUniform4ui", glProgramUniform4ui == NULL);
  glewInfoFunc(fi, "glProgramUniform4uiv", glProgramUniform4uiv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2dv", glProgramUniformMatrix2dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2fv", glProgramUniformMatrix2fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2x3dv", glProgramUniformMatrix2x3dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2x3fv", glProgramUniformMatrix2x3fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2x4dv", glProgramUniformMatrix2x4dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2x4fv", glProgramUniformMatrix2x4fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3dv", glProgramUniformMatrix3dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3fv", glProgramUniformMatrix3fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3x2dv", glProgramUniformMatrix3x2dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3x2fv", glProgramUniformMatrix3x2fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3x4dv", glProgramUniformMatrix3x4dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3x4fv", glProgramUniformMatrix3x4fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4dv", glProgramUniformMatrix4dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4fv", glProgramUniformMatrix4fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4x2dv", glProgramUniformMatrix4x2dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4x2fv", glProgramUniformMatrix4x2fv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4x3dv", glProgramUniformMatrix4x3dv == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4x3fv", glProgramUniformMatrix4x3fv == NULL);
  glewInfoFunc(fi, "glUseProgramStages", glUseProgramStages == NULL);
  glewInfoFunc(fi, "glValidateProgramPipeline", glValidateProgramPipeline == NULL);
}

#endif /* GL_ARB_separate_shader_objects */

#ifdef GL_ARB_shader_atomic_counter_ops

static void _glewInfo_GL_ARB_shader_atomic_counter_ops (void)
{
  glewPrintExt("GL_ARB_shader_atomic_counter_ops", GLEW_ARB_shader_atomic_counter_ops, glewIsSupported("GL_ARB_shader_atomic_counter_ops"), glewGetExtension("GL_ARB_shader_atomic_counter_ops"));
}

#endif /* GL_ARB_shader_atomic_counter_ops */

#ifdef GL_ARB_shader_atomic_counters

static void _glewInfo_GL_ARB_shader_atomic_counters (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_shader_atomic_counters", GLEW_ARB_shader_atomic_counters, glewIsSupported("GL_ARB_shader_atomic_counters"), glewGetExtension("GL_ARB_shader_atomic_counters"));

  glewInfoFunc(fi, "glGetActiveAtomicCounterBufferiv", glGetActiveAtomicCounterBufferiv == NULL);
}

#endif /* GL_ARB_shader_atomic_counters */

#ifdef GL_ARB_shader_ballot

static void _glewInfo_GL_ARB_shader_ballot (void)
{
  glewPrintExt("GL_ARB_shader_ballot", GLEW_ARB_shader_ballot, glewIsSupported("GL_ARB_shader_ballot"), glewGetExtension("GL_ARB_shader_ballot"));
}

#endif /* GL_ARB_shader_ballot */

#ifdef GL_ARB_shader_bit_encoding

static void _glewInfo_GL_ARB_shader_bit_encoding (void)
{
  glewPrintExt("GL_ARB_shader_bit_encoding", GLEW_ARB_shader_bit_encoding, glewIsSupported("GL_ARB_shader_bit_encoding"), glewGetExtension("GL_ARB_shader_bit_encoding"));
}

#endif /* GL_ARB_shader_bit_encoding */

#ifdef GL_ARB_shader_clock

static void _glewInfo_GL_ARB_shader_clock (void)
{
  glewPrintExt("GL_ARB_shader_clock", GLEW_ARB_shader_clock, glewIsSupported("GL_ARB_shader_clock"), glewGetExtension("GL_ARB_shader_clock"));
}

#endif /* GL_ARB_shader_clock */

#ifdef GL_ARB_shader_draw_parameters

static void _glewInfo_GL_ARB_shader_draw_parameters (void)
{
  glewPrintExt("GL_ARB_shader_draw_parameters", GLEW_ARB_shader_draw_parameters, glewIsSupported("GL_ARB_shader_draw_parameters"), glewGetExtension("GL_ARB_shader_draw_parameters"));
}

#endif /* GL_ARB_shader_draw_parameters */

#ifdef GL_ARB_shader_group_vote

static void _glewInfo_GL_ARB_shader_group_vote (void)
{
  glewPrintExt("GL_ARB_shader_group_vote", GLEW_ARB_shader_group_vote, glewIsSupported("GL_ARB_shader_group_vote"), glewGetExtension("GL_ARB_shader_group_vote"));
}

#endif /* GL_ARB_shader_group_vote */

#ifdef GL_ARB_shader_image_load_store

static void _glewInfo_GL_ARB_shader_image_load_store (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_shader_image_load_store", GLEW_ARB_shader_image_load_store, glewIsSupported("GL_ARB_shader_image_load_store"), glewGetExtension("GL_ARB_shader_image_load_store"));

  glewInfoFunc(fi, "glBindImageTexture", glBindImageTexture == NULL);
  glewInfoFunc(fi, "glMemoryBarrier", glMemoryBarrier == NULL);
}

#endif /* GL_ARB_shader_image_load_store */

#ifdef GL_ARB_shader_image_size

static void _glewInfo_GL_ARB_shader_image_size (void)
{
  glewPrintExt("GL_ARB_shader_image_size", GLEW_ARB_shader_image_size, glewIsSupported("GL_ARB_shader_image_size"), glewGetExtension("GL_ARB_shader_image_size"));
}

#endif /* GL_ARB_shader_image_size */

#ifdef GL_ARB_shader_objects

static void _glewInfo_GL_ARB_shader_objects (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_shader_objects", GLEW_ARB_shader_objects, glewIsSupported("GL_ARB_shader_objects"), glewGetExtension("GL_ARB_shader_objects"));

  glewInfoFunc(fi, "glAttachObjectARB", glAttachObjectARB == NULL);
  glewInfoFunc(fi, "glCompileShaderARB", glCompileShaderARB == NULL);
  glewInfoFunc(fi, "glCreateProgramObjectARB", glCreateProgramObjectARB == NULL);
  glewInfoFunc(fi, "glCreateShaderObjectARB", glCreateShaderObjectARB == NULL);
  glewInfoFunc(fi, "glDeleteObjectARB", glDeleteObjectARB == NULL);
  glewInfoFunc(fi, "glDetachObjectARB", glDetachObjectARB == NULL);
  glewInfoFunc(fi, "glGetActiveUniformARB", glGetActiveUniformARB == NULL);
  glewInfoFunc(fi, "glGetAttachedObjectsARB", glGetAttachedObjectsARB == NULL);
  glewInfoFunc(fi, "glGetHandleARB", glGetHandleARB == NULL);
  glewInfoFunc(fi, "glGetInfoLogARB", glGetInfoLogARB == NULL);
  glewInfoFunc(fi, "glGetObjectParameterfvARB", glGetObjectParameterfvARB == NULL);
  glewInfoFunc(fi, "glGetObjectParameterivARB", glGetObjectParameterivARB == NULL);
  glewInfoFunc(fi, "glGetShaderSourceARB", glGetShaderSourceARB == NULL);
  glewInfoFunc(fi, "glGetUniformLocationARB", glGetUniformLocationARB == NULL);
  glewInfoFunc(fi, "glGetUniformfvARB", glGetUniformfvARB == NULL);
  glewInfoFunc(fi, "glGetUniformivARB", glGetUniformivARB == NULL);
  glewInfoFunc(fi, "glLinkProgramARB", glLinkProgramARB == NULL);
  glewInfoFunc(fi, "glShaderSourceARB", glShaderSourceARB == NULL);
  glewInfoFunc(fi, "glUniform1fARB", glUniform1fARB == NULL);
  glewInfoFunc(fi, "glUniform1fvARB", glUniform1fvARB == NULL);
  glewInfoFunc(fi, "glUniform1iARB", glUniform1iARB == NULL);
  glewInfoFunc(fi, "glUniform1ivARB", glUniform1ivARB == NULL);
  glewInfoFunc(fi, "glUniform2fARB", glUniform2fARB == NULL);
  glewInfoFunc(fi, "glUniform2fvARB", glUniform2fvARB == NULL);
  glewInfoFunc(fi, "glUniform2iARB", glUniform2iARB == NULL);
  glewInfoFunc(fi, "glUniform2ivARB", glUniform2ivARB == NULL);
  glewInfoFunc(fi, "glUniform3fARB", glUniform3fARB == NULL);
  glewInfoFunc(fi, "glUniform3fvARB", glUniform3fvARB == NULL);
  glewInfoFunc(fi, "glUniform3iARB", glUniform3iARB == NULL);
  glewInfoFunc(fi, "glUniform3ivARB", glUniform3ivARB == NULL);
  glewInfoFunc(fi, "glUniform4fARB", glUniform4fARB == NULL);
  glewInfoFunc(fi, "glUniform4fvARB", glUniform4fvARB == NULL);
  glewInfoFunc(fi, "glUniform4iARB", glUniform4iARB == NULL);
  glewInfoFunc(fi, "glUniform4ivARB", glUniform4ivARB == NULL);
  glewInfoFunc(fi, "glUniformMatrix2fvARB", glUniformMatrix2fvARB == NULL);
  glewInfoFunc(fi, "glUniformMatrix3fvARB", glUniformMatrix3fvARB == NULL);
  glewInfoFunc(fi, "glUniformMatrix4fvARB", glUniformMatrix4fvARB == NULL);
  glewInfoFunc(fi, "glUseProgramObjectARB", glUseProgramObjectARB == NULL);
  glewInfoFunc(fi, "glValidateProgramARB", glValidateProgramARB == NULL);
}

#endif /* GL_ARB_shader_objects */

#ifdef GL_ARB_shader_precision

static void _glewInfo_GL_ARB_shader_precision (void)
{
  glewPrintExt("GL_ARB_shader_precision", GLEW_ARB_shader_precision, glewIsSupported("GL_ARB_shader_precision"), glewGetExtension("GL_ARB_shader_precision"));
}

#endif /* GL_ARB_shader_precision */

#ifdef GL_ARB_shader_stencil_export

static void _glewInfo_GL_ARB_shader_stencil_export (void)
{
  glewPrintExt("GL_ARB_shader_stencil_export", GLEW_ARB_shader_stencil_export, glewIsSupported("GL_ARB_shader_stencil_export"), glewGetExtension("GL_ARB_shader_stencil_export"));
}

#endif /* GL_ARB_shader_stencil_export */

#ifdef GL_ARB_shader_storage_buffer_object

static void _glewInfo_GL_ARB_shader_storage_buffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_shader_storage_buffer_object", GLEW_ARB_shader_storage_buffer_object, glewIsSupported("GL_ARB_shader_storage_buffer_object"), glewGetExtension("GL_ARB_shader_storage_buffer_object"));

  glewInfoFunc(fi, "glShaderStorageBlockBinding", glShaderStorageBlockBinding == NULL);
}

#endif /* GL_ARB_shader_storage_buffer_object */

#ifdef GL_ARB_shader_subroutine

static void _glewInfo_GL_ARB_shader_subroutine (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_shader_subroutine", GLEW_ARB_shader_subroutine, glewIsSupported("GL_ARB_shader_subroutine"), glewGetExtension("GL_ARB_shader_subroutine"));

  glewInfoFunc(fi, "glGetActiveSubroutineName", glGetActiveSubroutineName == NULL);
  glewInfoFunc(fi, "glGetActiveSubroutineUniformName", glGetActiveSubroutineUniformName == NULL);
  glewInfoFunc(fi, "glGetActiveSubroutineUniformiv", glGetActiveSubroutineUniformiv == NULL);
  glewInfoFunc(fi, "glGetProgramStageiv", glGetProgramStageiv == NULL);
  glewInfoFunc(fi, "glGetSubroutineIndex", glGetSubroutineIndex == NULL);
  glewInfoFunc(fi, "glGetSubroutineUniformLocation", glGetSubroutineUniformLocation == NULL);
  glewInfoFunc(fi, "glGetUniformSubroutineuiv", glGetUniformSubroutineuiv == NULL);
  glewInfoFunc(fi, "glUniformSubroutinesuiv", glUniformSubroutinesuiv == NULL);
}

#endif /* GL_ARB_shader_subroutine */

#ifdef GL_ARB_shader_texture_image_samples

static void _glewInfo_GL_ARB_shader_texture_image_samples (void)
{
  glewPrintExt("GL_ARB_shader_texture_image_samples", GLEW_ARB_shader_texture_image_samples, glewIsSupported("GL_ARB_shader_texture_image_samples"), glewGetExtension("GL_ARB_shader_texture_image_samples"));
}

#endif /* GL_ARB_shader_texture_image_samples */

#ifdef GL_ARB_shader_texture_lod

static void _glewInfo_GL_ARB_shader_texture_lod (void)
{
  glewPrintExt("GL_ARB_shader_texture_lod", GLEW_ARB_shader_texture_lod, glewIsSupported("GL_ARB_shader_texture_lod"), glewGetExtension("GL_ARB_shader_texture_lod"));
}

#endif /* GL_ARB_shader_texture_lod */

#ifdef GL_ARB_shader_viewport_layer_array

static void _glewInfo_GL_ARB_shader_viewport_layer_array (void)
{
  glewPrintExt("GL_ARB_shader_viewport_layer_array", GLEW_ARB_shader_viewport_layer_array, glewIsSupported("GL_ARB_shader_viewport_layer_array"), glewGetExtension("GL_ARB_shader_viewport_layer_array"));
}

#endif /* GL_ARB_shader_viewport_layer_array */

#ifdef GL_ARB_shading_language_100

static void _glewInfo_GL_ARB_shading_language_100 (void)
{
  glewPrintExt("GL_ARB_shading_language_100", GLEW_ARB_shading_language_100, glewIsSupported("GL_ARB_shading_language_100"), glewGetExtension("GL_ARB_shading_language_100"));
}

#endif /* GL_ARB_shading_language_100 */

#ifdef GL_ARB_shading_language_420pack

static void _glewInfo_GL_ARB_shading_language_420pack (void)
{
  glewPrintExt("GL_ARB_shading_language_420pack", GLEW_ARB_shading_language_420pack, glewIsSupported("GL_ARB_shading_language_420pack"), glewGetExtension("GL_ARB_shading_language_420pack"));
}

#endif /* GL_ARB_shading_language_420pack */

#ifdef GL_ARB_shading_language_include

static void _glewInfo_GL_ARB_shading_language_include (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_shading_language_include", GLEW_ARB_shading_language_include, glewIsSupported("GL_ARB_shading_language_include"), glewGetExtension("GL_ARB_shading_language_include"));

  glewInfoFunc(fi, "glCompileShaderIncludeARB", glCompileShaderIncludeARB == NULL);
  glewInfoFunc(fi, "glDeleteNamedStringARB", glDeleteNamedStringARB == NULL);
  glewInfoFunc(fi, "glGetNamedStringARB", glGetNamedStringARB == NULL);
  glewInfoFunc(fi, "glGetNamedStringivARB", glGetNamedStringivARB == NULL);
  glewInfoFunc(fi, "glIsNamedStringARB", glIsNamedStringARB == NULL);
  glewInfoFunc(fi, "glNamedStringARB", glNamedStringARB == NULL);
}

#endif /* GL_ARB_shading_language_include */

#ifdef GL_ARB_shading_language_packing

static void _glewInfo_GL_ARB_shading_language_packing (void)
{
  glewPrintExt("GL_ARB_shading_language_packing", GLEW_ARB_shading_language_packing, glewIsSupported("GL_ARB_shading_language_packing"), glewGetExtension("GL_ARB_shading_language_packing"));
}

#endif /* GL_ARB_shading_language_packing */

#ifdef GL_ARB_shadow

static void _glewInfo_GL_ARB_shadow (void)
{
  glewPrintExt("GL_ARB_shadow", GLEW_ARB_shadow, glewIsSupported("GL_ARB_shadow"), glewGetExtension("GL_ARB_shadow"));
}

#endif /* GL_ARB_shadow */

#ifdef GL_ARB_shadow_ambient

static void _glewInfo_GL_ARB_shadow_ambient (void)
{
  glewPrintExt("GL_ARB_shadow_ambient", GLEW_ARB_shadow_ambient, glewIsSupported("GL_ARB_shadow_ambient"), glewGetExtension("GL_ARB_shadow_ambient"));
}

#endif /* GL_ARB_shadow_ambient */

#ifdef GL_ARB_sparse_buffer

static void _glewInfo_GL_ARB_sparse_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_sparse_buffer", GLEW_ARB_sparse_buffer, glewIsSupported("GL_ARB_sparse_buffer"), glewGetExtension("GL_ARB_sparse_buffer"));

  glewInfoFunc(fi, "glBufferPageCommitmentARB", glBufferPageCommitmentARB == NULL);
}

#endif /* GL_ARB_sparse_buffer */

#ifdef GL_ARB_sparse_texture

static void _glewInfo_GL_ARB_sparse_texture (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_sparse_texture", GLEW_ARB_sparse_texture, glewIsSupported("GL_ARB_sparse_texture"), glewGetExtension("GL_ARB_sparse_texture"));

  glewInfoFunc(fi, "glTexPageCommitmentARB", glTexPageCommitmentARB == NULL);
}

#endif /* GL_ARB_sparse_texture */

#ifdef GL_ARB_sparse_texture2

static void _glewInfo_GL_ARB_sparse_texture2 (void)
{
  glewPrintExt("GL_ARB_sparse_texture2", GLEW_ARB_sparse_texture2, glewIsSupported("GL_ARB_sparse_texture2"), glewGetExtension("GL_ARB_sparse_texture2"));
}

#endif /* GL_ARB_sparse_texture2 */

#ifdef GL_ARB_sparse_texture_clamp

static void _glewInfo_GL_ARB_sparse_texture_clamp (void)
{
  glewPrintExt("GL_ARB_sparse_texture_clamp", GLEW_ARB_sparse_texture_clamp, glewIsSupported("GL_ARB_sparse_texture_clamp"), glewGetExtension("GL_ARB_sparse_texture_clamp"));
}

#endif /* GL_ARB_sparse_texture_clamp */

#ifdef GL_ARB_spirv_extensions

static void _glewInfo_GL_ARB_spirv_extensions (void)
{
  glewPrintExt("GL_ARB_spirv_extensions", GLEW_ARB_spirv_extensions, glewIsSupported("GL_ARB_spirv_extensions"), glewGetExtension("GL_ARB_spirv_extensions"));
}

#endif /* GL_ARB_spirv_extensions */

#ifdef GL_ARB_stencil_texturing

static void _glewInfo_GL_ARB_stencil_texturing (void)
{
  glewPrintExt("GL_ARB_stencil_texturing", GLEW_ARB_stencil_texturing, glewIsSupported("GL_ARB_stencil_texturing"), glewGetExtension("GL_ARB_stencil_texturing"));
}

#endif /* GL_ARB_stencil_texturing */

#ifdef GL_ARB_sync

static void _glewInfo_GL_ARB_sync (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_sync", GLEW_ARB_sync, glewIsSupported("GL_ARB_sync"), glewGetExtension("GL_ARB_sync"));

  glewInfoFunc(fi, "glClientWaitSync", glClientWaitSync == NULL);
  glewInfoFunc(fi, "glDeleteSync", glDeleteSync == NULL);
  glewInfoFunc(fi, "glFenceSync", glFenceSync == NULL);
  glewInfoFunc(fi, "glGetInteger64v", glGetInteger64v == NULL);
  glewInfoFunc(fi, "glGetSynciv", glGetSynciv == NULL);
  glewInfoFunc(fi, "glIsSync", glIsSync == NULL);
  glewInfoFunc(fi, "glWaitSync", glWaitSync == NULL);
}

#endif /* GL_ARB_sync */

#ifdef GL_ARB_tessellation_shader

static void _glewInfo_GL_ARB_tessellation_shader (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_tessellation_shader", GLEW_ARB_tessellation_shader, glewIsSupported("GL_ARB_tessellation_shader"), glewGetExtension("GL_ARB_tessellation_shader"));

  glewInfoFunc(fi, "glPatchParameterfv", glPatchParameterfv == NULL);
  glewInfoFunc(fi, "glPatchParameteri", glPatchParameteri == NULL);
}

#endif /* GL_ARB_tessellation_shader */

#ifdef GL_ARB_texture_barrier

static void _glewInfo_GL_ARB_texture_barrier (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_barrier", GLEW_ARB_texture_barrier, glewIsSupported("GL_ARB_texture_barrier"), glewGetExtension("GL_ARB_texture_barrier"));

  glewInfoFunc(fi, "glTextureBarrier", glTextureBarrier == NULL);
}

#endif /* GL_ARB_texture_barrier */

#ifdef GL_ARB_texture_border_clamp

static void _glewInfo_GL_ARB_texture_border_clamp (void)
{
  glewPrintExt("GL_ARB_texture_border_clamp", GLEW_ARB_texture_border_clamp, glewIsSupported("GL_ARB_texture_border_clamp"), glewGetExtension("GL_ARB_texture_border_clamp"));
}

#endif /* GL_ARB_texture_border_clamp */

#ifdef GL_ARB_texture_buffer_object

static void _glewInfo_GL_ARB_texture_buffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_buffer_object", GLEW_ARB_texture_buffer_object, glewIsSupported("GL_ARB_texture_buffer_object"), glewGetExtension("GL_ARB_texture_buffer_object"));

  glewInfoFunc(fi, "glTexBufferARB", glTexBufferARB == NULL);
}

#endif /* GL_ARB_texture_buffer_object */

#ifdef GL_ARB_texture_buffer_object_rgb32

static void _glewInfo_GL_ARB_texture_buffer_object_rgb32 (void)
{
  glewPrintExt("GL_ARB_texture_buffer_object_rgb32", GLEW_ARB_texture_buffer_object_rgb32, glewIsSupported("GL_ARB_texture_buffer_object_rgb32"), glewGetExtension("GL_ARB_texture_buffer_object_rgb32"));
}

#endif /* GL_ARB_texture_buffer_object_rgb32 */

#ifdef GL_ARB_texture_buffer_range

static void _glewInfo_GL_ARB_texture_buffer_range (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_buffer_range", GLEW_ARB_texture_buffer_range, glewIsSupported("GL_ARB_texture_buffer_range"), glewGetExtension("GL_ARB_texture_buffer_range"));

  glewInfoFunc(fi, "glTexBufferRange", glTexBufferRange == NULL);
  glewInfoFunc(fi, "glTextureBufferRangeEXT", glTextureBufferRangeEXT == NULL);
}

#endif /* GL_ARB_texture_buffer_range */

#ifdef GL_ARB_texture_compression

static void _glewInfo_GL_ARB_texture_compression (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_compression", GLEW_ARB_texture_compression, glewIsSupported("GL_ARB_texture_compression"), glewGetExtension("GL_ARB_texture_compression"));

  glewInfoFunc(fi, "glCompressedTexImage1DARB", glCompressedTexImage1DARB == NULL);
  glewInfoFunc(fi, "glCompressedTexImage2DARB", glCompressedTexImage2DARB == NULL);
  glewInfoFunc(fi, "glCompressedTexImage3DARB", glCompressedTexImage3DARB == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage1DARB", glCompressedTexSubImage1DARB == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage2DARB", glCompressedTexSubImage2DARB == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage3DARB", glCompressedTexSubImage3DARB == NULL);
  glewInfoFunc(fi, "glGetCompressedTexImageARB", glGetCompressedTexImageARB == NULL);
}

#endif /* GL_ARB_texture_compression */

#ifdef GL_ARB_texture_compression_bptc

static void _glewInfo_GL_ARB_texture_compression_bptc (void)
{
  glewPrintExt("GL_ARB_texture_compression_bptc", GLEW_ARB_texture_compression_bptc, glewIsSupported("GL_ARB_texture_compression_bptc"), glewGetExtension("GL_ARB_texture_compression_bptc"));
}

#endif /* GL_ARB_texture_compression_bptc */

#ifdef GL_ARB_texture_compression_rgtc

static void _glewInfo_GL_ARB_texture_compression_rgtc (void)
{
  glewPrintExt("GL_ARB_texture_compression_rgtc", GLEW_ARB_texture_compression_rgtc, glewIsSupported("GL_ARB_texture_compression_rgtc"), glewGetExtension("GL_ARB_texture_compression_rgtc"));
}

#endif /* GL_ARB_texture_compression_rgtc */

#ifdef GL_ARB_texture_cube_map

static void _glewInfo_GL_ARB_texture_cube_map (void)
{
  glewPrintExt("GL_ARB_texture_cube_map", GLEW_ARB_texture_cube_map, glewIsSupported("GL_ARB_texture_cube_map"), glewGetExtension("GL_ARB_texture_cube_map"));
}

#endif /* GL_ARB_texture_cube_map */

#ifdef GL_ARB_texture_cube_map_array

static void _glewInfo_GL_ARB_texture_cube_map_array (void)
{
  glewPrintExt("GL_ARB_texture_cube_map_array", GLEW_ARB_texture_cube_map_array, glewIsSupported("GL_ARB_texture_cube_map_array"), glewGetExtension("GL_ARB_texture_cube_map_array"));
}

#endif /* GL_ARB_texture_cube_map_array */

#ifdef GL_ARB_texture_env_add

static void _glewInfo_GL_ARB_texture_env_add (void)
{
  glewPrintExt("GL_ARB_texture_env_add", GLEW_ARB_texture_env_add, glewIsSupported("GL_ARB_texture_env_add"), glewGetExtension("GL_ARB_texture_env_add"));
}

#endif /* GL_ARB_texture_env_add */

#ifdef GL_ARB_texture_env_combine

static void _glewInfo_GL_ARB_texture_env_combine (void)
{
  glewPrintExt("GL_ARB_texture_env_combine", GLEW_ARB_texture_env_combine, glewIsSupported("GL_ARB_texture_env_combine"), glewGetExtension("GL_ARB_texture_env_combine"));
}

#endif /* GL_ARB_texture_env_combine */

#ifdef GL_ARB_texture_env_crossbar

static void _glewInfo_GL_ARB_texture_env_crossbar (void)
{
  glewPrintExt("GL_ARB_texture_env_crossbar", GLEW_ARB_texture_env_crossbar, glewIsSupported("GL_ARB_texture_env_crossbar"), glewGetExtension("GL_ARB_texture_env_crossbar"));
}

#endif /* GL_ARB_texture_env_crossbar */

#ifdef GL_ARB_texture_env_dot3

static void _glewInfo_GL_ARB_texture_env_dot3 (void)
{
  glewPrintExt("GL_ARB_texture_env_dot3", GLEW_ARB_texture_env_dot3, glewIsSupported("GL_ARB_texture_env_dot3"), glewGetExtension("GL_ARB_texture_env_dot3"));
}

#endif /* GL_ARB_texture_env_dot3 */

#ifdef GL_ARB_texture_filter_anisotropic

static void _glewInfo_GL_ARB_texture_filter_anisotropic (void)
{
  glewPrintExt("GL_ARB_texture_filter_anisotropic", GLEW_ARB_texture_filter_anisotropic, glewIsSupported("GL_ARB_texture_filter_anisotropic"), glewGetExtension("GL_ARB_texture_filter_anisotropic"));
}

#endif /* GL_ARB_texture_filter_anisotropic */

#ifdef GL_ARB_texture_filter_minmax

static void _glewInfo_GL_ARB_texture_filter_minmax (void)
{
  glewPrintExt("GL_ARB_texture_filter_minmax", GLEW_ARB_texture_filter_minmax, glewIsSupported("GL_ARB_texture_filter_minmax"), glewGetExtension("GL_ARB_texture_filter_minmax"));
}

#endif /* GL_ARB_texture_filter_minmax */

#ifdef GL_ARB_texture_float

static void _glewInfo_GL_ARB_texture_float (void)
{
  glewPrintExt("GL_ARB_texture_float", GLEW_ARB_texture_float, glewIsSupported("GL_ARB_texture_float"), glewGetExtension("GL_ARB_texture_float"));
}

#endif /* GL_ARB_texture_float */

#ifdef GL_ARB_texture_gather

static void _glewInfo_GL_ARB_texture_gather (void)
{
  glewPrintExt("GL_ARB_texture_gather", GLEW_ARB_texture_gather, glewIsSupported("GL_ARB_texture_gather"), glewGetExtension("GL_ARB_texture_gather"));
}

#endif /* GL_ARB_texture_gather */

#ifdef GL_ARB_texture_mirror_clamp_to_edge

static void _glewInfo_GL_ARB_texture_mirror_clamp_to_edge (void)
{
  glewPrintExt("GL_ARB_texture_mirror_clamp_to_edge", GLEW_ARB_texture_mirror_clamp_to_edge, glewIsSupported("GL_ARB_texture_mirror_clamp_to_edge"), glewGetExtension("GL_ARB_texture_mirror_clamp_to_edge"));
}

#endif /* GL_ARB_texture_mirror_clamp_to_edge */

#ifdef GL_ARB_texture_mirrored_repeat

static void _glewInfo_GL_ARB_texture_mirrored_repeat (void)
{
  glewPrintExt("GL_ARB_texture_mirrored_repeat", GLEW_ARB_texture_mirrored_repeat, glewIsSupported("GL_ARB_texture_mirrored_repeat"), glewGetExtension("GL_ARB_texture_mirrored_repeat"));
}

#endif /* GL_ARB_texture_mirrored_repeat */

#ifdef GL_ARB_texture_multisample

static void _glewInfo_GL_ARB_texture_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_multisample", GLEW_ARB_texture_multisample, glewIsSupported("GL_ARB_texture_multisample"), glewGetExtension("GL_ARB_texture_multisample"));

  glewInfoFunc(fi, "glGetMultisamplefv", glGetMultisamplefv == NULL);
  glewInfoFunc(fi, "glSampleMaski", glSampleMaski == NULL);
  glewInfoFunc(fi, "glTexImage2DMultisample", glTexImage2DMultisample == NULL);
  glewInfoFunc(fi, "glTexImage3DMultisample", glTexImage3DMultisample == NULL);
}

#endif /* GL_ARB_texture_multisample */

#ifdef GL_ARB_texture_non_power_of_two

static void _glewInfo_GL_ARB_texture_non_power_of_two (void)
{
  glewPrintExt("GL_ARB_texture_non_power_of_two", GLEW_ARB_texture_non_power_of_two, glewIsSupported("GL_ARB_texture_non_power_of_two"), glewGetExtension("GL_ARB_texture_non_power_of_two"));
}

#endif /* GL_ARB_texture_non_power_of_two */

#ifdef GL_ARB_texture_query_levels

static void _glewInfo_GL_ARB_texture_query_levels (void)
{
  glewPrintExt("GL_ARB_texture_query_levels", GLEW_ARB_texture_query_levels, glewIsSupported("GL_ARB_texture_query_levels"), glewGetExtension("GL_ARB_texture_query_levels"));
}

#endif /* GL_ARB_texture_query_levels */

#ifdef GL_ARB_texture_query_lod

static void _glewInfo_GL_ARB_texture_query_lod (void)
{
  glewPrintExt("GL_ARB_texture_query_lod", GLEW_ARB_texture_query_lod, glewIsSupported("GL_ARB_texture_query_lod"), glewGetExtension("GL_ARB_texture_query_lod"));
}

#endif /* GL_ARB_texture_query_lod */

#ifdef GL_ARB_texture_rectangle

static void _glewInfo_GL_ARB_texture_rectangle (void)
{
  glewPrintExt("GL_ARB_texture_rectangle", GLEW_ARB_texture_rectangle, glewIsSupported("GL_ARB_texture_rectangle"), glewGetExtension("GL_ARB_texture_rectangle"));
}

#endif /* GL_ARB_texture_rectangle */

#ifdef GL_ARB_texture_rg

static void _glewInfo_GL_ARB_texture_rg (void)
{
  glewPrintExt("GL_ARB_texture_rg", GLEW_ARB_texture_rg, glewIsSupported("GL_ARB_texture_rg"), glewGetExtension("GL_ARB_texture_rg"));
}

#endif /* GL_ARB_texture_rg */

#ifdef GL_ARB_texture_rgb10_a2ui

static void _glewInfo_GL_ARB_texture_rgb10_a2ui (void)
{
  glewPrintExt("GL_ARB_texture_rgb10_a2ui", GLEW_ARB_texture_rgb10_a2ui, glewIsSupported("GL_ARB_texture_rgb10_a2ui"), glewGetExtension("GL_ARB_texture_rgb10_a2ui"));
}

#endif /* GL_ARB_texture_rgb10_a2ui */

#ifdef GL_ARB_texture_stencil8

static void _glewInfo_GL_ARB_texture_stencil8 (void)
{
  glewPrintExt("GL_ARB_texture_stencil8", GLEW_ARB_texture_stencil8, glewIsSupported("GL_ARB_texture_stencil8"), glewGetExtension("GL_ARB_texture_stencil8"));
}

#endif /* GL_ARB_texture_stencil8 */

#ifdef GL_ARB_texture_storage

static void _glewInfo_GL_ARB_texture_storage (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_storage", GLEW_ARB_texture_storage, glewIsSupported("GL_ARB_texture_storage"), glewGetExtension("GL_ARB_texture_storage"));

  glewInfoFunc(fi, "glTexStorage1D", glTexStorage1D == NULL);
  glewInfoFunc(fi, "glTexStorage2D", glTexStorage2D == NULL);
  glewInfoFunc(fi, "glTexStorage3D", glTexStorage3D == NULL);
}

#endif /* GL_ARB_texture_storage */

#ifdef GL_ARB_texture_storage_multisample

static void _glewInfo_GL_ARB_texture_storage_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_storage_multisample", GLEW_ARB_texture_storage_multisample, glewIsSupported("GL_ARB_texture_storage_multisample"), glewGetExtension("GL_ARB_texture_storage_multisample"));

  glewInfoFunc(fi, "glTexStorage2DMultisample", glTexStorage2DMultisample == NULL);
  glewInfoFunc(fi, "glTexStorage3DMultisample", glTexStorage3DMultisample == NULL);
  glewInfoFunc(fi, "glTextureStorage2DMultisampleEXT", glTextureStorage2DMultisampleEXT == NULL);
  glewInfoFunc(fi, "glTextureStorage3DMultisampleEXT", glTextureStorage3DMultisampleEXT == NULL);
}

#endif /* GL_ARB_texture_storage_multisample */

#ifdef GL_ARB_texture_swizzle

static void _glewInfo_GL_ARB_texture_swizzle (void)
{
  glewPrintExt("GL_ARB_texture_swizzle", GLEW_ARB_texture_swizzle, glewIsSupported("GL_ARB_texture_swizzle"), glewGetExtension("GL_ARB_texture_swizzle"));
}

#endif /* GL_ARB_texture_swizzle */

#ifdef GL_ARB_texture_view

static void _glewInfo_GL_ARB_texture_view (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_texture_view", GLEW_ARB_texture_view, glewIsSupported("GL_ARB_texture_view"), glewGetExtension("GL_ARB_texture_view"));

  glewInfoFunc(fi, "glTextureView", glTextureView == NULL);
}

#endif /* GL_ARB_texture_view */

#ifdef GL_ARB_timer_query

static void _glewInfo_GL_ARB_timer_query (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_timer_query", GLEW_ARB_timer_query, glewIsSupported("GL_ARB_timer_query"), glewGetExtension("GL_ARB_timer_query"));

  glewInfoFunc(fi, "glGetQueryObjecti64v", glGetQueryObjecti64v == NULL);
  glewInfoFunc(fi, "glGetQueryObjectui64v", glGetQueryObjectui64v == NULL);
  glewInfoFunc(fi, "glQueryCounter", glQueryCounter == NULL);
}

#endif /* GL_ARB_timer_query */

#ifdef GL_ARB_transform_feedback2

static void _glewInfo_GL_ARB_transform_feedback2 (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_transform_feedback2", GLEW_ARB_transform_feedback2, glewIsSupported("GL_ARB_transform_feedback2"), glewGetExtension("GL_ARB_transform_feedback2"));

  glewInfoFunc(fi, "glBindTransformFeedback", glBindTransformFeedback == NULL);
  glewInfoFunc(fi, "glDeleteTransformFeedbacks", glDeleteTransformFeedbacks == NULL);
  glewInfoFunc(fi, "glDrawTransformFeedback", glDrawTransformFeedback == NULL);
  glewInfoFunc(fi, "glGenTransformFeedbacks", glGenTransformFeedbacks == NULL);
  glewInfoFunc(fi, "glIsTransformFeedback", glIsTransformFeedback == NULL);
  glewInfoFunc(fi, "glPauseTransformFeedback", glPauseTransformFeedback == NULL);
  glewInfoFunc(fi, "glResumeTransformFeedback", glResumeTransformFeedback == NULL);
}

#endif /* GL_ARB_transform_feedback2 */

#ifdef GL_ARB_transform_feedback3

static void _glewInfo_GL_ARB_transform_feedback3 (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_transform_feedback3", GLEW_ARB_transform_feedback3, glewIsSupported("GL_ARB_transform_feedback3"), glewGetExtension("GL_ARB_transform_feedback3"));

  glewInfoFunc(fi, "glBeginQueryIndexed", glBeginQueryIndexed == NULL);
  glewInfoFunc(fi, "glDrawTransformFeedbackStream", glDrawTransformFeedbackStream == NULL);
  glewInfoFunc(fi, "glEndQueryIndexed", glEndQueryIndexed == NULL);
  glewInfoFunc(fi, "glGetQueryIndexediv", glGetQueryIndexediv == NULL);
}

#endif /* GL_ARB_transform_feedback3 */

#ifdef GL_ARB_transform_feedback_instanced

static void _glewInfo_GL_ARB_transform_feedback_instanced (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_transform_feedback_instanced", GLEW_ARB_transform_feedback_instanced, glewIsSupported("GL_ARB_transform_feedback_instanced"), glewGetExtension("GL_ARB_transform_feedback_instanced"));

  glewInfoFunc(fi, "glDrawTransformFeedbackInstanced", glDrawTransformFeedbackInstanced == NULL);
  glewInfoFunc(fi, "glDrawTransformFeedbackStreamInstanced", glDrawTransformFeedbackStreamInstanced == NULL);
}

#endif /* GL_ARB_transform_feedback_instanced */

#ifdef GL_ARB_transform_feedback_overflow_query

static void _glewInfo_GL_ARB_transform_feedback_overflow_query (void)
{
  glewPrintExt("GL_ARB_transform_feedback_overflow_query", GLEW_ARB_transform_feedback_overflow_query, glewIsSupported("GL_ARB_transform_feedback_overflow_query"), glewGetExtension("GL_ARB_transform_feedback_overflow_query"));
}

#endif /* GL_ARB_transform_feedback_overflow_query */

#ifdef GL_ARB_transpose_matrix

static void _glewInfo_GL_ARB_transpose_matrix (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_transpose_matrix", GLEW_ARB_transpose_matrix, glewIsSupported("GL_ARB_transpose_matrix"), glewGetExtension("GL_ARB_transpose_matrix"));

  glewInfoFunc(fi, "glLoadTransposeMatrixdARB", glLoadTransposeMatrixdARB == NULL);
  glewInfoFunc(fi, "glLoadTransposeMatrixfARB", glLoadTransposeMatrixfARB == NULL);
  glewInfoFunc(fi, "glMultTransposeMatrixdARB", glMultTransposeMatrixdARB == NULL);
  glewInfoFunc(fi, "glMultTransposeMatrixfARB", glMultTransposeMatrixfARB == NULL);
}

#endif /* GL_ARB_transpose_matrix */

#ifdef GL_ARB_uniform_buffer_object

static void _glewInfo_GL_ARB_uniform_buffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_uniform_buffer_object", GLEW_ARB_uniform_buffer_object, glewIsSupported("GL_ARB_uniform_buffer_object"), glewGetExtension("GL_ARB_uniform_buffer_object"));

  glewInfoFunc(fi, "glBindBufferBase", glBindBufferBase == NULL);
  glewInfoFunc(fi, "glBindBufferRange", glBindBufferRange == NULL);
  glewInfoFunc(fi, "glGetActiveUniformBlockName", glGetActiveUniformBlockName == NULL);
  glewInfoFunc(fi, "glGetActiveUniformBlockiv", glGetActiveUniformBlockiv == NULL);
  glewInfoFunc(fi, "glGetActiveUniformName", glGetActiveUniformName == NULL);
  glewInfoFunc(fi, "glGetActiveUniformsiv", glGetActiveUniformsiv == NULL);
  glewInfoFunc(fi, "glGetIntegeri_v", glGetIntegeri_v == NULL);
  glewInfoFunc(fi, "glGetUniformBlockIndex", glGetUniformBlockIndex == NULL);
  glewInfoFunc(fi, "glGetUniformIndices", glGetUniformIndices == NULL);
  glewInfoFunc(fi, "glUniformBlockBinding", glUniformBlockBinding == NULL);
}

#endif /* GL_ARB_uniform_buffer_object */

#ifdef GL_ARB_vertex_array_bgra

static void _glewInfo_GL_ARB_vertex_array_bgra (void)
{
  glewPrintExt("GL_ARB_vertex_array_bgra", GLEW_ARB_vertex_array_bgra, glewIsSupported("GL_ARB_vertex_array_bgra"), glewGetExtension("GL_ARB_vertex_array_bgra"));
}

#endif /* GL_ARB_vertex_array_bgra */

#ifdef GL_ARB_vertex_array_object

static void _glewInfo_GL_ARB_vertex_array_object (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_array_object", GLEW_ARB_vertex_array_object, glewIsSupported("GL_ARB_vertex_array_object"), glewGetExtension("GL_ARB_vertex_array_object"));

  glewInfoFunc(fi, "glBindVertexArray", glBindVertexArray == NULL);
  glewInfoFunc(fi, "glDeleteVertexArrays", glDeleteVertexArrays == NULL);
  glewInfoFunc(fi, "glGenVertexArrays", glGenVertexArrays == NULL);
  glewInfoFunc(fi, "glIsVertexArray", glIsVertexArray == NULL);
}

#endif /* GL_ARB_vertex_array_object */

#ifdef GL_ARB_vertex_attrib_64bit

static void _glewInfo_GL_ARB_vertex_attrib_64bit (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_attrib_64bit", GLEW_ARB_vertex_attrib_64bit, glewIsSupported("GL_ARB_vertex_attrib_64bit"), glewGetExtension("GL_ARB_vertex_attrib_64bit"));

  glewInfoFunc(fi, "glGetVertexAttribLdv", glGetVertexAttribLdv == NULL);
  glewInfoFunc(fi, "glVertexAttribL1d", glVertexAttribL1d == NULL);
  glewInfoFunc(fi, "glVertexAttribL1dv", glVertexAttribL1dv == NULL);
  glewInfoFunc(fi, "glVertexAttribL2d", glVertexAttribL2d == NULL);
  glewInfoFunc(fi, "glVertexAttribL2dv", glVertexAttribL2dv == NULL);
  glewInfoFunc(fi, "glVertexAttribL3d", glVertexAttribL3d == NULL);
  glewInfoFunc(fi, "glVertexAttribL3dv", glVertexAttribL3dv == NULL);
  glewInfoFunc(fi, "glVertexAttribL4d", glVertexAttribL4d == NULL);
  glewInfoFunc(fi, "glVertexAttribL4dv", glVertexAttribL4dv == NULL);
  glewInfoFunc(fi, "glVertexAttribLPointer", glVertexAttribLPointer == NULL);
}

#endif /* GL_ARB_vertex_attrib_64bit */

#ifdef GL_ARB_vertex_attrib_binding

static void _glewInfo_GL_ARB_vertex_attrib_binding (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_attrib_binding", GLEW_ARB_vertex_attrib_binding, glewIsSupported("GL_ARB_vertex_attrib_binding"), glewGetExtension("GL_ARB_vertex_attrib_binding"));

  glewInfoFunc(fi, "glBindVertexBuffer", glBindVertexBuffer == NULL);
  glewInfoFunc(fi, "glVertexArrayBindVertexBufferEXT", glVertexArrayBindVertexBufferEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribBindingEXT", glVertexArrayVertexAttribBindingEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribFormatEXT", glVertexArrayVertexAttribFormatEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribIFormatEXT", glVertexArrayVertexAttribIFormatEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribLFormatEXT", glVertexArrayVertexAttribLFormatEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexBindingDivisorEXT", glVertexArrayVertexBindingDivisorEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribBinding", glVertexAttribBinding == NULL);
  glewInfoFunc(fi, "glVertexAttribFormat", glVertexAttribFormat == NULL);
  glewInfoFunc(fi, "glVertexAttribIFormat", glVertexAttribIFormat == NULL);
  glewInfoFunc(fi, "glVertexAttribLFormat", glVertexAttribLFormat == NULL);
  glewInfoFunc(fi, "glVertexBindingDivisor", glVertexBindingDivisor == NULL);
}

#endif /* GL_ARB_vertex_attrib_binding */

#ifdef GL_ARB_vertex_blend

static void _glewInfo_GL_ARB_vertex_blend (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_blend", GLEW_ARB_vertex_blend, glewIsSupported("GL_ARB_vertex_blend"), glewGetExtension("GL_ARB_vertex_blend"));

  glewInfoFunc(fi, "glVertexBlendARB", glVertexBlendARB == NULL);
  glewInfoFunc(fi, "glWeightPointerARB", glWeightPointerARB == NULL);
  glewInfoFunc(fi, "glWeightbvARB", glWeightbvARB == NULL);
  glewInfoFunc(fi, "glWeightdvARB", glWeightdvARB == NULL);
  glewInfoFunc(fi, "glWeightfvARB", glWeightfvARB == NULL);
  glewInfoFunc(fi, "glWeightivARB", glWeightivARB == NULL);
  glewInfoFunc(fi, "glWeightsvARB", glWeightsvARB == NULL);
  glewInfoFunc(fi, "glWeightubvARB", glWeightubvARB == NULL);
  glewInfoFunc(fi, "glWeightuivARB", glWeightuivARB == NULL);
  glewInfoFunc(fi, "glWeightusvARB", glWeightusvARB == NULL);
}

#endif /* GL_ARB_vertex_blend */

#ifdef GL_ARB_vertex_buffer_object

static void _glewInfo_GL_ARB_vertex_buffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_buffer_object", GLEW_ARB_vertex_buffer_object, glewIsSupported("GL_ARB_vertex_buffer_object"), glewGetExtension("GL_ARB_vertex_buffer_object"));

  glewInfoFunc(fi, "glBindBufferARB", glBindBufferARB == NULL);
  glewInfoFunc(fi, "glBufferDataARB", glBufferDataARB == NULL);
  glewInfoFunc(fi, "glBufferSubDataARB", glBufferSubDataARB == NULL);
  glewInfoFunc(fi, "glDeleteBuffersARB", glDeleteBuffersARB == NULL);
  glewInfoFunc(fi, "glGenBuffersARB", glGenBuffersARB == NULL);
  glewInfoFunc(fi, "glGetBufferParameterivARB", glGetBufferParameterivARB == NULL);
  glewInfoFunc(fi, "glGetBufferPointervARB", glGetBufferPointervARB == NULL);
  glewInfoFunc(fi, "glGetBufferSubDataARB", glGetBufferSubDataARB == NULL);
  glewInfoFunc(fi, "glIsBufferARB", glIsBufferARB == NULL);
  glewInfoFunc(fi, "glMapBufferARB", glMapBufferARB == NULL);
  glewInfoFunc(fi, "glUnmapBufferARB", glUnmapBufferARB == NULL);
}

#endif /* GL_ARB_vertex_buffer_object */

#ifdef GL_ARB_vertex_program

static void _glewInfo_GL_ARB_vertex_program (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_program", GLEW_ARB_vertex_program, glewIsSupported("GL_ARB_vertex_program"), glewGetExtension("GL_ARB_vertex_program"));

  glewInfoFunc(fi, "glBindProgramARB", glBindProgramARB == NULL);
  glewInfoFunc(fi, "glDeleteProgramsARB", glDeleteProgramsARB == NULL);
  glewInfoFunc(fi, "glDisableVertexAttribArrayARB", glDisableVertexAttribArrayARB == NULL);
  glewInfoFunc(fi, "glEnableVertexAttribArrayARB", glEnableVertexAttribArrayARB == NULL);
  glewInfoFunc(fi, "glGenProgramsARB", glGenProgramsARB == NULL);
  glewInfoFunc(fi, "glGetProgramEnvParameterdvARB", glGetProgramEnvParameterdvARB == NULL);
  glewInfoFunc(fi, "glGetProgramEnvParameterfvARB", glGetProgramEnvParameterfvARB == NULL);
  glewInfoFunc(fi, "glGetProgramLocalParameterdvARB", glGetProgramLocalParameterdvARB == NULL);
  glewInfoFunc(fi, "glGetProgramLocalParameterfvARB", glGetProgramLocalParameterfvARB == NULL);
  glewInfoFunc(fi, "glGetProgramStringARB", glGetProgramStringARB == NULL);
  glewInfoFunc(fi, "glGetProgramivARB", glGetProgramivARB == NULL);
  glewInfoFunc(fi, "glGetVertexAttribPointervARB", glGetVertexAttribPointervARB == NULL);
  glewInfoFunc(fi, "glGetVertexAttribdvARB", glGetVertexAttribdvARB == NULL);
  glewInfoFunc(fi, "glGetVertexAttribfvARB", glGetVertexAttribfvARB == NULL);
  glewInfoFunc(fi, "glGetVertexAttribivARB", glGetVertexAttribivARB == NULL);
  glewInfoFunc(fi, "glIsProgramARB", glIsProgramARB == NULL);
  glewInfoFunc(fi, "glProgramEnvParameter4dARB", glProgramEnvParameter4dARB == NULL);
  glewInfoFunc(fi, "glProgramEnvParameter4dvARB", glProgramEnvParameter4dvARB == NULL);
  glewInfoFunc(fi, "glProgramEnvParameter4fARB", glProgramEnvParameter4fARB == NULL);
  glewInfoFunc(fi, "glProgramEnvParameter4fvARB", glProgramEnvParameter4fvARB == NULL);
  glewInfoFunc(fi, "glProgramLocalParameter4dARB", glProgramLocalParameter4dARB == NULL);
  glewInfoFunc(fi, "glProgramLocalParameter4dvARB", glProgramLocalParameter4dvARB == NULL);
  glewInfoFunc(fi, "glProgramLocalParameter4fARB", glProgramLocalParameter4fARB == NULL);
  glewInfoFunc(fi, "glProgramLocalParameter4fvARB", glProgramLocalParameter4fvARB == NULL);
  glewInfoFunc(fi, "glProgramStringARB", glProgramStringARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib1dARB", glVertexAttrib1dARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib1dvARB", glVertexAttrib1dvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib1fARB", glVertexAttrib1fARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib1fvARB", glVertexAttrib1fvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib1sARB", glVertexAttrib1sARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib1svARB", glVertexAttrib1svARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib2dARB", glVertexAttrib2dARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib2dvARB", glVertexAttrib2dvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib2fARB", glVertexAttrib2fARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib2fvARB", glVertexAttrib2fvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib2sARB", glVertexAttrib2sARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib2svARB", glVertexAttrib2svARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib3dARB", glVertexAttrib3dARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib3dvARB", glVertexAttrib3dvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib3fARB", glVertexAttrib3fARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib3fvARB", glVertexAttrib3fvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib3sARB", glVertexAttrib3sARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib3svARB", glVertexAttrib3svARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4NbvARB", glVertexAttrib4NbvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4NivARB", glVertexAttrib4NivARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4NsvARB", glVertexAttrib4NsvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4NubARB", glVertexAttrib4NubARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4NubvARB", glVertexAttrib4NubvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4NuivARB", glVertexAttrib4NuivARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4NusvARB", glVertexAttrib4NusvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4bvARB", glVertexAttrib4bvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4dARB", glVertexAttrib4dARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4dvARB", glVertexAttrib4dvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4fARB", glVertexAttrib4fARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4fvARB", glVertexAttrib4fvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4ivARB", glVertexAttrib4ivARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4sARB", glVertexAttrib4sARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4svARB", glVertexAttrib4svARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4ubvARB", glVertexAttrib4ubvARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4uivARB", glVertexAttrib4uivARB == NULL);
  glewInfoFunc(fi, "glVertexAttrib4usvARB", glVertexAttrib4usvARB == NULL);
  glewInfoFunc(fi, "glVertexAttribPointerARB", glVertexAttribPointerARB == NULL);
}

#endif /* GL_ARB_vertex_program */

#ifdef GL_ARB_vertex_shader

static void _glewInfo_GL_ARB_vertex_shader (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_shader", GLEW_ARB_vertex_shader, glewIsSupported("GL_ARB_vertex_shader"), glewGetExtension("GL_ARB_vertex_shader"));

  glewInfoFunc(fi, "glBindAttribLocationARB", glBindAttribLocationARB == NULL);
  glewInfoFunc(fi, "glGetActiveAttribARB", glGetActiveAttribARB == NULL);
  glewInfoFunc(fi, "glGetAttribLocationARB", glGetAttribLocationARB == NULL);
}

#endif /* GL_ARB_vertex_shader */

#ifdef GL_ARB_vertex_type_10f_11f_11f_rev

static void _glewInfo_GL_ARB_vertex_type_10f_11f_11f_rev (void)
{
  glewPrintExt("GL_ARB_vertex_type_10f_11f_11f_rev", GLEW_ARB_vertex_type_10f_11f_11f_rev, glewIsSupported("GL_ARB_vertex_type_10f_11f_11f_rev"), glewGetExtension("GL_ARB_vertex_type_10f_11f_11f_rev"));
}

#endif /* GL_ARB_vertex_type_10f_11f_11f_rev */

#ifdef GL_ARB_vertex_type_2_10_10_10_rev

static void _glewInfo_GL_ARB_vertex_type_2_10_10_10_rev (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_vertex_type_2_10_10_10_rev", GLEW_ARB_vertex_type_2_10_10_10_rev, glewIsSupported("GL_ARB_vertex_type_2_10_10_10_rev"), glewGetExtension("GL_ARB_vertex_type_2_10_10_10_rev"));

  glewInfoFunc(fi, "glColorP3ui", glColorP3ui == NULL);
  glewInfoFunc(fi, "glColorP3uiv", glColorP3uiv == NULL);
  glewInfoFunc(fi, "glColorP4ui", glColorP4ui == NULL);
  glewInfoFunc(fi, "glColorP4uiv", glColorP4uiv == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP1ui", glMultiTexCoordP1ui == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP1uiv", glMultiTexCoordP1uiv == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP2ui", glMultiTexCoordP2ui == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP2uiv", glMultiTexCoordP2uiv == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP3ui", glMultiTexCoordP3ui == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP3uiv", glMultiTexCoordP3uiv == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP4ui", glMultiTexCoordP4ui == NULL);
  glewInfoFunc(fi, "glMultiTexCoordP4uiv", glMultiTexCoordP4uiv == NULL);
  glewInfoFunc(fi, "glNormalP3ui", glNormalP3ui == NULL);
  glewInfoFunc(fi, "glNormalP3uiv", glNormalP3uiv == NULL);
  glewInfoFunc(fi, "glSecondaryColorP3ui", glSecondaryColorP3ui == NULL);
  glewInfoFunc(fi, "glSecondaryColorP3uiv", glSecondaryColorP3uiv == NULL);
  glewInfoFunc(fi, "glTexCoordP1ui", glTexCoordP1ui == NULL);
  glewInfoFunc(fi, "glTexCoordP1uiv", glTexCoordP1uiv == NULL);
  glewInfoFunc(fi, "glTexCoordP2ui", glTexCoordP2ui == NULL);
  glewInfoFunc(fi, "glTexCoordP2uiv", glTexCoordP2uiv == NULL);
  glewInfoFunc(fi, "glTexCoordP3ui", glTexCoordP3ui == NULL);
  glewInfoFunc(fi, "glTexCoordP3uiv", glTexCoordP3uiv == NULL);
  glewInfoFunc(fi, "glTexCoordP4ui", glTexCoordP4ui == NULL);
  glewInfoFunc(fi, "glTexCoordP4uiv", glTexCoordP4uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribP1ui", glVertexAttribP1ui == NULL);
  glewInfoFunc(fi, "glVertexAttribP1uiv", glVertexAttribP1uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribP2ui", glVertexAttribP2ui == NULL);
  glewInfoFunc(fi, "glVertexAttribP2uiv", glVertexAttribP2uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribP3ui", glVertexAttribP3ui == NULL);
  glewInfoFunc(fi, "glVertexAttribP3uiv", glVertexAttribP3uiv == NULL);
  glewInfoFunc(fi, "glVertexAttribP4ui", glVertexAttribP4ui == NULL);
  glewInfoFunc(fi, "glVertexAttribP4uiv", glVertexAttribP4uiv == NULL);
  glewInfoFunc(fi, "glVertexP2ui", glVertexP2ui == NULL);
  glewInfoFunc(fi, "glVertexP2uiv", glVertexP2uiv == NULL);
  glewInfoFunc(fi, "glVertexP3ui", glVertexP3ui == NULL);
  glewInfoFunc(fi, "glVertexP3uiv", glVertexP3uiv == NULL);
  glewInfoFunc(fi, "glVertexP4ui", glVertexP4ui == NULL);
  glewInfoFunc(fi, "glVertexP4uiv", glVertexP4uiv == NULL);
}

#endif /* GL_ARB_vertex_type_2_10_10_10_rev */

#ifdef GL_ARB_viewport_array

static void _glewInfo_GL_ARB_viewport_array (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_viewport_array", GLEW_ARB_viewport_array, glewIsSupported("GL_ARB_viewport_array"), glewGetExtension("GL_ARB_viewport_array"));

  glewInfoFunc(fi, "glDepthRangeArrayv", glDepthRangeArrayv == NULL);
  glewInfoFunc(fi, "glDepthRangeIndexed", glDepthRangeIndexed == NULL);
  glewInfoFunc(fi, "glGetDoublei_v", glGetDoublei_v == NULL);
  glewInfoFunc(fi, "glGetFloati_v", glGetFloati_v == NULL);
  glewInfoFunc(fi, "glScissorArrayv", glScissorArrayv == NULL);
  glewInfoFunc(fi, "glScissorIndexed", glScissorIndexed == NULL);
  glewInfoFunc(fi, "glScissorIndexedv", glScissorIndexedv == NULL);
  glewInfoFunc(fi, "glViewportArrayv", glViewportArrayv == NULL);
  glewInfoFunc(fi, "glViewportIndexedf", glViewportIndexedf == NULL);
  glewInfoFunc(fi, "glViewportIndexedfv", glViewportIndexedfv == NULL);
}

#endif /* GL_ARB_viewport_array */

#ifdef GL_ARB_window_pos

static void _glewInfo_GL_ARB_window_pos (void)
{
  GLboolean fi = glewPrintExt("GL_ARB_window_pos", GLEW_ARB_window_pos, glewIsSupported("GL_ARB_window_pos"), glewGetExtension("GL_ARB_window_pos"));

  glewInfoFunc(fi, "glWindowPos2dARB", glWindowPos2dARB == NULL);
  glewInfoFunc(fi, "glWindowPos2dvARB", glWindowPos2dvARB == NULL);
  glewInfoFunc(fi, "glWindowPos2fARB", glWindowPos2fARB == NULL);
  glewInfoFunc(fi, "glWindowPos2fvARB", glWindowPos2fvARB == NULL);
  glewInfoFunc(fi, "glWindowPos2iARB", glWindowPos2iARB == NULL);
  glewInfoFunc(fi, "glWindowPos2ivARB", glWindowPos2ivARB == NULL);
  glewInfoFunc(fi, "glWindowPos2sARB", glWindowPos2sARB == NULL);
  glewInfoFunc(fi, "glWindowPos2svARB", glWindowPos2svARB == NULL);
  glewInfoFunc(fi, "glWindowPos3dARB", glWindowPos3dARB == NULL);
  glewInfoFunc(fi, "glWindowPos3dvARB", glWindowPos3dvARB == NULL);
  glewInfoFunc(fi, "glWindowPos3fARB", glWindowPos3fARB == NULL);
  glewInfoFunc(fi, "glWindowPos3fvARB", glWindowPos3fvARB == NULL);
  glewInfoFunc(fi, "glWindowPos3iARB", glWindowPos3iARB == NULL);
  glewInfoFunc(fi, "glWindowPos3ivARB", glWindowPos3ivARB == NULL);
  glewInfoFunc(fi, "glWindowPos3sARB", glWindowPos3sARB == NULL);
  glewInfoFunc(fi, "glWindowPos3svARB", glWindowPos3svARB == NULL);
}

#endif /* GL_ARB_window_pos */

#ifdef GL_ARM_mali_program_binary

static void _glewInfo_GL_ARM_mali_program_binary (void)
{
  glewPrintExt("GL_ARM_mali_program_binary", GLEW_ARM_mali_program_binary, glewIsSupported("GL_ARM_mali_program_binary"), glewGetExtension("GL_ARM_mali_program_binary"));
}

#endif /* GL_ARM_mali_program_binary */

#ifdef GL_ARM_mali_shader_binary

static void _glewInfo_GL_ARM_mali_shader_binary (void)
{
  glewPrintExt("GL_ARM_mali_shader_binary", GLEW_ARM_mali_shader_binary, glewIsSupported("GL_ARM_mali_shader_binary"), glewGetExtension("GL_ARM_mali_shader_binary"));
}

#endif /* GL_ARM_mali_shader_binary */

#ifdef GL_ARM_rgba8

static void _glewInfo_GL_ARM_rgba8 (void)
{
  glewPrintExt("GL_ARM_rgba8", GLEW_ARM_rgba8, glewIsSupported("GL_ARM_rgba8"), glewGetExtension("GL_ARM_rgba8"));
}

#endif /* GL_ARM_rgba8 */

#ifdef GL_ARM_shader_framebuffer_fetch

static void _glewInfo_GL_ARM_shader_framebuffer_fetch (void)
{
  glewPrintExt("GL_ARM_shader_framebuffer_fetch", GLEW_ARM_shader_framebuffer_fetch, glewIsSupported("GL_ARM_shader_framebuffer_fetch"), glewGetExtension("GL_ARM_shader_framebuffer_fetch"));
}

#endif /* GL_ARM_shader_framebuffer_fetch */

#ifdef GL_ARM_shader_framebuffer_fetch_depth_stencil

static void _glewInfo_GL_ARM_shader_framebuffer_fetch_depth_stencil (void)
{
  glewPrintExt("GL_ARM_shader_framebuffer_fetch_depth_stencil", GLEW_ARM_shader_framebuffer_fetch_depth_stencil, glewIsSupported("GL_ARM_shader_framebuffer_fetch_depth_stencil"), glewGetExtension("GL_ARM_shader_framebuffer_fetch_depth_stencil"));
}

#endif /* GL_ARM_shader_framebuffer_fetch_depth_stencil */

#ifdef GL_ARM_texture_unnormalized_coordinates

static void _glewInfo_GL_ARM_texture_unnormalized_coordinates (void)
{
  glewPrintExt("GL_ARM_texture_unnormalized_coordinates", GLEW_ARM_texture_unnormalized_coordinates, glewIsSupported("GL_ARM_texture_unnormalized_coordinates"), glewGetExtension("GL_ARM_texture_unnormalized_coordinates"));
}

#endif /* GL_ARM_texture_unnormalized_coordinates */

#ifdef GL_ATIX_point_sprites

static void _glewInfo_GL_ATIX_point_sprites (void)
{
  glewPrintExt("GL_ATIX_point_sprites", GLEW_ATIX_point_sprites, glewIsSupported("GL_ATIX_point_sprites"), glewGetExtension("GL_ATIX_point_sprites"));
}

#endif /* GL_ATIX_point_sprites */

#ifdef GL_ATIX_texture_env_combine3

static void _glewInfo_GL_ATIX_texture_env_combine3 (void)
{
  glewPrintExt("GL_ATIX_texture_env_combine3", GLEW_ATIX_texture_env_combine3, glewIsSupported("GL_ATIX_texture_env_combine3"), glewGetExtension("GL_ATIX_texture_env_combine3"));
}

#endif /* GL_ATIX_texture_env_combine3 */

#ifdef GL_ATIX_texture_env_route

static void _glewInfo_GL_ATIX_texture_env_route (void)
{
  glewPrintExt("GL_ATIX_texture_env_route", GLEW_ATIX_texture_env_route, glewIsSupported("GL_ATIX_texture_env_route"), glewGetExtension("GL_ATIX_texture_env_route"));
}

#endif /* GL_ATIX_texture_env_route */

#ifdef GL_ATIX_vertex_shader_output_point_size

static void _glewInfo_GL_ATIX_vertex_shader_output_point_size (void)
{
  glewPrintExt("GL_ATIX_vertex_shader_output_point_size", GLEW_ATIX_vertex_shader_output_point_size, glewIsSupported("GL_ATIX_vertex_shader_output_point_size"), glewGetExtension("GL_ATIX_vertex_shader_output_point_size"));
}

#endif /* GL_ATIX_vertex_shader_output_point_size */

#ifdef GL_ATI_draw_buffers

static void _glewInfo_GL_ATI_draw_buffers (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_draw_buffers", GLEW_ATI_draw_buffers, glewIsSupported("GL_ATI_draw_buffers"), glewGetExtension("GL_ATI_draw_buffers"));

  glewInfoFunc(fi, "glDrawBuffersATI", glDrawBuffersATI == NULL);
}

#endif /* GL_ATI_draw_buffers */

#ifdef GL_ATI_element_array

static void _glewInfo_GL_ATI_element_array (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_element_array", GLEW_ATI_element_array, glewIsSupported("GL_ATI_element_array"), glewGetExtension("GL_ATI_element_array"));

  glewInfoFunc(fi, "glDrawElementArrayATI", glDrawElementArrayATI == NULL);
  glewInfoFunc(fi, "glDrawRangeElementArrayATI", glDrawRangeElementArrayATI == NULL);
  glewInfoFunc(fi, "glElementPointerATI", glElementPointerATI == NULL);
}

#endif /* GL_ATI_element_array */

#ifdef GL_ATI_envmap_bumpmap

static void _glewInfo_GL_ATI_envmap_bumpmap (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_envmap_bumpmap", GLEW_ATI_envmap_bumpmap, glewIsSupported("GL_ATI_envmap_bumpmap"), glewGetExtension("GL_ATI_envmap_bumpmap"));

  glewInfoFunc(fi, "glGetTexBumpParameterfvATI", glGetTexBumpParameterfvATI == NULL);
  glewInfoFunc(fi, "glGetTexBumpParameterivATI", glGetTexBumpParameterivATI == NULL);
  glewInfoFunc(fi, "glTexBumpParameterfvATI", glTexBumpParameterfvATI == NULL);
  glewInfoFunc(fi, "glTexBumpParameterivATI", glTexBumpParameterivATI == NULL);
}

#endif /* GL_ATI_envmap_bumpmap */

#ifdef GL_ATI_fragment_shader

static void _glewInfo_GL_ATI_fragment_shader (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_fragment_shader", GLEW_ATI_fragment_shader, glewIsSupported("GL_ATI_fragment_shader"), glewGetExtension("GL_ATI_fragment_shader"));

  glewInfoFunc(fi, "glAlphaFragmentOp1ATI", glAlphaFragmentOp1ATI == NULL);
  glewInfoFunc(fi, "glAlphaFragmentOp2ATI", glAlphaFragmentOp2ATI == NULL);
  glewInfoFunc(fi, "glAlphaFragmentOp3ATI", glAlphaFragmentOp3ATI == NULL);
  glewInfoFunc(fi, "glBeginFragmentShaderATI", glBeginFragmentShaderATI == NULL);
  glewInfoFunc(fi, "glBindFragmentShaderATI", glBindFragmentShaderATI == NULL);
  glewInfoFunc(fi, "glColorFragmentOp1ATI", glColorFragmentOp1ATI == NULL);
  glewInfoFunc(fi, "glColorFragmentOp2ATI", glColorFragmentOp2ATI == NULL);
  glewInfoFunc(fi, "glColorFragmentOp3ATI", glColorFragmentOp3ATI == NULL);
  glewInfoFunc(fi, "glDeleteFragmentShaderATI", glDeleteFragmentShaderATI == NULL);
  glewInfoFunc(fi, "glEndFragmentShaderATI", glEndFragmentShaderATI == NULL);
  glewInfoFunc(fi, "glGenFragmentShadersATI", glGenFragmentShadersATI == NULL);
  glewInfoFunc(fi, "glPassTexCoordATI", glPassTexCoordATI == NULL);
  glewInfoFunc(fi, "glSampleMapATI", glSampleMapATI == NULL);
  glewInfoFunc(fi, "glSetFragmentShaderConstantATI", glSetFragmentShaderConstantATI == NULL);
}

#endif /* GL_ATI_fragment_shader */

#ifdef GL_ATI_map_object_buffer

static void _glewInfo_GL_ATI_map_object_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_map_object_buffer", GLEW_ATI_map_object_buffer, glewIsSupported("GL_ATI_map_object_buffer"), glewGetExtension("GL_ATI_map_object_buffer"));

  glewInfoFunc(fi, "glMapObjectBufferATI", glMapObjectBufferATI == NULL);
  glewInfoFunc(fi, "glUnmapObjectBufferATI", glUnmapObjectBufferATI == NULL);
}

#endif /* GL_ATI_map_object_buffer */

#ifdef GL_ATI_meminfo

static void _glewInfo_GL_ATI_meminfo (void)
{
  glewPrintExt("GL_ATI_meminfo", GLEW_ATI_meminfo, glewIsSupported("GL_ATI_meminfo"), glewGetExtension("GL_ATI_meminfo"));
}

#endif /* GL_ATI_meminfo */

#ifdef GL_ATI_pn_triangles

static void _glewInfo_GL_ATI_pn_triangles (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_pn_triangles", GLEW_ATI_pn_triangles, glewIsSupported("GL_ATI_pn_triangles"), glewGetExtension("GL_ATI_pn_triangles"));

  glewInfoFunc(fi, "glPNTrianglesfATI", glPNTrianglesfATI == NULL);
  glewInfoFunc(fi, "glPNTrianglesiATI", glPNTrianglesiATI == NULL);
}

#endif /* GL_ATI_pn_triangles */

#ifdef GL_ATI_separate_stencil

static void _glewInfo_GL_ATI_separate_stencil (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_separate_stencil", GLEW_ATI_separate_stencil, glewIsSupported("GL_ATI_separate_stencil"), glewGetExtension("GL_ATI_separate_stencil"));

  glewInfoFunc(fi, "glStencilFuncSeparateATI", glStencilFuncSeparateATI == NULL);
  glewInfoFunc(fi, "glStencilOpSeparateATI", glStencilOpSeparateATI == NULL);
}

#endif /* GL_ATI_separate_stencil */

#ifdef GL_ATI_shader_texture_lod

static void _glewInfo_GL_ATI_shader_texture_lod (void)
{
  glewPrintExt("GL_ATI_shader_texture_lod", GLEW_ATI_shader_texture_lod, glewIsSupported("GL_ATI_shader_texture_lod"), glewGetExtension("GL_ATI_shader_texture_lod"));
}

#endif /* GL_ATI_shader_texture_lod */

#ifdef GL_ATI_text_fragment_shader

static void _glewInfo_GL_ATI_text_fragment_shader (void)
{
  glewPrintExt("GL_ATI_text_fragment_shader", GLEW_ATI_text_fragment_shader, glewIsSupported("GL_ATI_text_fragment_shader"), glewGetExtension("GL_ATI_text_fragment_shader"));
}

#endif /* GL_ATI_text_fragment_shader */

#ifdef GL_ATI_texture_compression_3dc

static void _glewInfo_GL_ATI_texture_compression_3dc (void)
{
  glewPrintExt("GL_ATI_texture_compression_3dc", GLEW_ATI_texture_compression_3dc, glewIsSupported("GL_ATI_texture_compression_3dc"), glewGetExtension("GL_ATI_texture_compression_3dc"));
}

#endif /* GL_ATI_texture_compression_3dc */

#ifdef GL_ATI_texture_env_combine3

static void _glewInfo_GL_ATI_texture_env_combine3 (void)
{
  glewPrintExt("GL_ATI_texture_env_combine3", GLEW_ATI_texture_env_combine3, glewIsSupported("GL_ATI_texture_env_combine3"), glewGetExtension("GL_ATI_texture_env_combine3"));
}

#endif /* GL_ATI_texture_env_combine3 */

#ifdef GL_ATI_texture_float

static void _glewInfo_GL_ATI_texture_float (void)
{
  glewPrintExt("GL_ATI_texture_float", GLEW_ATI_texture_float, glewIsSupported("GL_ATI_texture_float"), glewGetExtension("GL_ATI_texture_float"));
}

#endif /* GL_ATI_texture_float */

#ifdef GL_ATI_texture_mirror_once

static void _glewInfo_GL_ATI_texture_mirror_once (void)
{
  glewPrintExt("GL_ATI_texture_mirror_once", GLEW_ATI_texture_mirror_once, glewIsSupported("GL_ATI_texture_mirror_once"), glewGetExtension("GL_ATI_texture_mirror_once"));
}

#endif /* GL_ATI_texture_mirror_once */

#ifdef GL_ATI_vertex_array_object

static void _glewInfo_GL_ATI_vertex_array_object (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_vertex_array_object", GLEW_ATI_vertex_array_object, glewIsSupported("GL_ATI_vertex_array_object"), glewGetExtension("GL_ATI_vertex_array_object"));

  glewInfoFunc(fi, "glArrayObjectATI", glArrayObjectATI == NULL);
  glewInfoFunc(fi, "glFreeObjectBufferATI", glFreeObjectBufferATI == NULL);
  glewInfoFunc(fi, "glGetArrayObjectfvATI", glGetArrayObjectfvATI == NULL);
  glewInfoFunc(fi, "glGetArrayObjectivATI", glGetArrayObjectivATI == NULL);
  glewInfoFunc(fi, "glGetObjectBufferfvATI", glGetObjectBufferfvATI == NULL);
  glewInfoFunc(fi, "glGetObjectBufferivATI", glGetObjectBufferivATI == NULL);
  glewInfoFunc(fi, "glGetVariantArrayObjectfvATI", glGetVariantArrayObjectfvATI == NULL);
  glewInfoFunc(fi, "glGetVariantArrayObjectivATI", glGetVariantArrayObjectivATI == NULL);
  glewInfoFunc(fi, "glIsObjectBufferATI", glIsObjectBufferATI == NULL);
  glewInfoFunc(fi, "glNewObjectBufferATI", glNewObjectBufferATI == NULL);
  glewInfoFunc(fi, "glUpdateObjectBufferATI", glUpdateObjectBufferATI == NULL);
  glewInfoFunc(fi, "glVariantArrayObjectATI", glVariantArrayObjectATI == NULL);
}

#endif /* GL_ATI_vertex_array_object */

#ifdef GL_ATI_vertex_attrib_array_object

static void _glewInfo_GL_ATI_vertex_attrib_array_object (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_vertex_attrib_array_object", GLEW_ATI_vertex_attrib_array_object, glewIsSupported("GL_ATI_vertex_attrib_array_object"), glewGetExtension("GL_ATI_vertex_attrib_array_object"));

  glewInfoFunc(fi, "glGetVertexAttribArrayObjectfvATI", glGetVertexAttribArrayObjectfvATI == NULL);
  glewInfoFunc(fi, "glGetVertexAttribArrayObjectivATI", glGetVertexAttribArrayObjectivATI == NULL);
  glewInfoFunc(fi, "glVertexAttribArrayObjectATI", glVertexAttribArrayObjectATI == NULL);
}

#endif /* GL_ATI_vertex_attrib_array_object */

#ifdef GL_ATI_vertex_streams

static void _glewInfo_GL_ATI_vertex_streams (void)
{
  GLboolean fi = glewPrintExt("GL_ATI_vertex_streams", GLEW_ATI_vertex_streams, glewIsSupported("GL_ATI_vertex_streams"), glewGetExtension("GL_ATI_vertex_streams"));

  glewInfoFunc(fi, "glClientActiveVertexStreamATI", glClientActiveVertexStreamATI == NULL);
  glewInfoFunc(fi, "glNormalStream3bATI", glNormalStream3bATI == NULL);
  glewInfoFunc(fi, "glNormalStream3bvATI", glNormalStream3bvATI == NULL);
  glewInfoFunc(fi, "glNormalStream3dATI", glNormalStream3dATI == NULL);
  glewInfoFunc(fi, "glNormalStream3dvATI", glNormalStream3dvATI == NULL);
  glewInfoFunc(fi, "glNormalStream3fATI", glNormalStream3fATI == NULL);
  glewInfoFunc(fi, "glNormalStream3fvATI", glNormalStream3fvATI == NULL);
  glewInfoFunc(fi, "glNormalStream3iATI", glNormalStream3iATI == NULL);
  glewInfoFunc(fi, "glNormalStream3ivATI", glNormalStream3ivATI == NULL);
  glewInfoFunc(fi, "glNormalStream3sATI", glNormalStream3sATI == NULL);
  glewInfoFunc(fi, "glNormalStream3svATI", glNormalStream3svATI == NULL);
  glewInfoFunc(fi, "glVertexBlendEnvfATI", glVertexBlendEnvfATI == NULL);
  glewInfoFunc(fi, "glVertexBlendEnviATI", glVertexBlendEnviATI == NULL);
  glewInfoFunc(fi, "glVertexStream1dATI", glVertexStream1dATI == NULL);
  glewInfoFunc(fi, "glVertexStream1dvATI", glVertexStream1dvATI == NULL);
  glewInfoFunc(fi, "glVertexStream1fATI", glVertexStream1fATI == NULL);
  glewInfoFunc(fi, "glVertexStream1fvATI", glVertexStream1fvATI == NULL);
  glewInfoFunc(fi, "glVertexStream1iATI", glVertexStream1iATI == NULL);
  glewInfoFunc(fi, "glVertexStream1ivATI", glVertexStream1ivATI == NULL);
  glewInfoFunc(fi, "glVertexStream1sATI", glVertexStream1sATI == NULL);
  glewInfoFunc(fi, "glVertexStream1svATI", glVertexStream1svATI == NULL);
  glewInfoFunc(fi, "glVertexStream2dATI", glVertexStream2dATI == NULL);
  glewInfoFunc(fi, "glVertexStream2dvATI", glVertexStream2dvATI == NULL);
  glewInfoFunc(fi, "glVertexStream2fATI", glVertexStream2fATI == NULL);
  glewInfoFunc(fi, "glVertexStream2fvATI", glVertexStream2fvATI == NULL);
  glewInfoFunc(fi, "glVertexStream2iATI", glVertexStream2iATI == NULL);
  glewInfoFunc(fi, "glVertexStream2ivATI", glVertexStream2ivATI == NULL);
  glewInfoFunc(fi, "glVertexStream2sATI", glVertexStream2sATI == NULL);
  glewInfoFunc(fi, "glVertexStream2svATI", glVertexStream2svATI == NULL);
  glewInfoFunc(fi, "glVertexStream3dATI", glVertexStream3dATI == NULL);
  glewInfoFunc(fi, "glVertexStream3dvATI", glVertexStream3dvATI == NULL);
  glewInfoFunc(fi, "glVertexStream3fATI", glVertexStream3fATI == NULL);
  glewInfoFunc(fi, "glVertexStream3fvATI", glVertexStream3fvATI == NULL);
  glewInfoFunc(fi, "glVertexStream3iATI", glVertexStream3iATI == NULL);
  glewInfoFunc(fi, "glVertexStream3ivATI", glVertexStream3ivATI == NULL);
  glewInfoFunc(fi, "glVertexStream3sATI", glVertexStream3sATI == NULL);
  glewInfoFunc(fi, "glVertexStream3svATI", glVertexStream3svATI == NULL);
  glewInfoFunc(fi, "glVertexStream4dATI", glVertexStream4dATI == NULL);
  glewInfoFunc(fi, "glVertexStream4dvATI", glVertexStream4dvATI == NULL);
  glewInfoFunc(fi, "glVertexStream4fATI", glVertexStream4fATI == NULL);
  glewInfoFunc(fi, "glVertexStream4fvATI", glVertexStream4fvATI == NULL);
  glewInfoFunc(fi, "glVertexStream4iATI", glVertexStream4iATI == NULL);
  glewInfoFunc(fi, "glVertexStream4ivATI", glVertexStream4ivATI == NULL);
  glewInfoFunc(fi, "glVertexStream4sATI", glVertexStream4sATI == NULL);
  glewInfoFunc(fi, "glVertexStream4svATI", glVertexStream4svATI == NULL);
}

#endif /* GL_ATI_vertex_streams */

#ifdef GL_DMP_program_binary

static void _glewInfo_GL_DMP_program_binary (void)
{
  glewPrintExt("GL_DMP_program_binary", GLEW_DMP_program_binary, glewIsSupported("GL_DMP_program_binary"), glewGetExtension("GL_DMP_program_binary"));
}

#endif /* GL_DMP_program_binary */

#ifdef GL_DMP_shader_binary

static void _glewInfo_GL_DMP_shader_binary (void)
{
  glewPrintExt("GL_DMP_shader_binary", GLEW_DMP_shader_binary, glewIsSupported("GL_DMP_shader_binary"), glewGetExtension("GL_DMP_shader_binary"));
}

#endif /* GL_DMP_shader_binary */

#ifdef GL_EXT_422_pixels

static void _glewInfo_GL_EXT_422_pixels (void)
{
  glewPrintExt("GL_EXT_422_pixels", GLEW_EXT_422_pixels, glewIsSupported("GL_EXT_422_pixels"), glewGetExtension("GL_EXT_422_pixels"));
}

#endif /* GL_EXT_422_pixels */

#ifdef GL_EXT_Cg_shader

static void _glewInfo_GL_EXT_Cg_shader (void)
{
  glewPrintExt("GL_EXT_Cg_shader", GLEW_EXT_Cg_shader, glewIsSupported("GL_EXT_Cg_shader"), glewGetExtension("GL_EXT_Cg_shader"));
}

#endif /* GL_EXT_Cg_shader */

#ifdef GL_EXT_EGL_image_array

static void _glewInfo_GL_EXT_EGL_image_array (void)
{
  glewPrintExt("GL_EXT_EGL_image_array", GLEW_EXT_EGL_image_array, glewIsSupported("GL_EXT_EGL_image_array"), glewGetExtension("GL_EXT_EGL_image_array"));
}

#endif /* GL_EXT_EGL_image_array */

#ifdef GL_EXT_EGL_image_external_wrap_modes

static void _glewInfo_GL_EXT_EGL_image_external_wrap_modes (void)
{
  glewPrintExt("GL_EXT_EGL_image_external_wrap_modes", GLEW_EXT_EGL_image_external_wrap_modes, glewIsSupported("GL_EXT_EGL_image_external_wrap_modes"), glewGetExtension("GL_EXT_EGL_image_external_wrap_modes"));
}

#endif /* GL_EXT_EGL_image_external_wrap_modes */

#ifdef GL_EXT_EGL_image_storage

static void _glewInfo_GL_EXT_EGL_image_storage (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_EGL_image_storage", GLEW_EXT_EGL_image_storage, glewIsSupported("GL_EXT_EGL_image_storage"), glewGetExtension("GL_EXT_EGL_image_storage"));

  glewInfoFunc(fi, "glEGLImageTargetTexStorageEXT", glEGLImageTargetTexStorageEXT == NULL);
  glewInfoFunc(fi, "glEGLImageTargetTextureStorageEXT", glEGLImageTargetTextureStorageEXT == NULL);
}

#endif /* GL_EXT_EGL_image_storage */

#ifdef GL_EXT_EGL_sync

static void _glewInfo_GL_EXT_EGL_sync (void)
{
  glewPrintExt("GL_EXT_EGL_sync", GLEW_EXT_EGL_sync, glewIsSupported("GL_EXT_EGL_sync"), glewGetExtension("GL_EXT_EGL_sync"));
}

#endif /* GL_EXT_EGL_sync */

#ifdef GL_EXT_YUV_target

static void _glewInfo_GL_EXT_YUV_target (void)
{
  glewPrintExt("GL_EXT_YUV_target", GLEW_EXT_YUV_target, glewIsSupported("GL_EXT_YUV_target"), glewGetExtension("GL_EXT_YUV_target"));
}

#endif /* GL_EXT_YUV_target */

#ifdef GL_EXT_abgr

static void _glewInfo_GL_EXT_abgr (void)
{
  glewPrintExt("GL_EXT_abgr", GLEW_EXT_abgr, glewIsSupported("GL_EXT_abgr"), glewGetExtension("GL_EXT_abgr"));
}

#endif /* GL_EXT_abgr */

#ifdef GL_EXT_base_instance

static void _glewInfo_GL_EXT_base_instance (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_base_instance", GLEW_EXT_base_instance, glewIsSupported("GL_EXT_base_instance"), glewGetExtension("GL_EXT_base_instance"));

  glewInfoFunc(fi, "glDrawArraysInstancedBaseInstanceEXT", glDrawArraysInstancedBaseInstanceEXT == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedBaseInstanceEXT", glDrawElementsInstancedBaseInstanceEXT == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedBaseVertexBaseInstanceEXT", glDrawElementsInstancedBaseVertexBaseInstanceEXT == NULL);
}

#endif /* GL_EXT_base_instance */

#ifdef GL_EXT_bgra

static void _glewInfo_GL_EXT_bgra (void)
{
  glewPrintExt("GL_EXT_bgra", GLEW_EXT_bgra, glewIsSupported("GL_EXT_bgra"), glewGetExtension("GL_EXT_bgra"));
}

#endif /* GL_EXT_bgra */

#ifdef GL_EXT_bindable_uniform

static void _glewInfo_GL_EXT_bindable_uniform (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_bindable_uniform", GLEW_EXT_bindable_uniform, glewIsSupported("GL_EXT_bindable_uniform"), glewGetExtension("GL_EXT_bindable_uniform"));

  glewInfoFunc(fi, "glGetUniformBufferSizeEXT", glGetUniformBufferSizeEXT == NULL);
  glewInfoFunc(fi, "glGetUniformOffsetEXT", glGetUniformOffsetEXT == NULL);
  glewInfoFunc(fi, "glUniformBufferEXT", glUniformBufferEXT == NULL);
}

#endif /* GL_EXT_bindable_uniform */

#ifdef GL_EXT_blend_color

static void _glewInfo_GL_EXT_blend_color (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_blend_color", GLEW_EXT_blend_color, glewIsSupported("GL_EXT_blend_color"), glewGetExtension("GL_EXT_blend_color"));

  glewInfoFunc(fi, "glBlendColorEXT", glBlendColorEXT == NULL);
}

#endif /* GL_EXT_blend_color */

#ifdef GL_EXT_blend_equation_separate

static void _glewInfo_GL_EXT_blend_equation_separate (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_blend_equation_separate", GLEW_EXT_blend_equation_separate, glewIsSupported("GL_EXT_blend_equation_separate"), glewGetExtension("GL_EXT_blend_equation_separate"));

  glewInfoFunc(fi, "glBlendEquationSeparateEXT", glBlendEquationSeparateEXT == NULL);
}

#endif /* GL_EXT_blend_equation_separate */

#ifdef GL_EXT_blend_func_extended

static void _glewInfo_GL_EXT_blend_func_extended (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_blend_func_extended", GLEW_EXT_blend_func_extended, glewIsSupported("GL_EXT_blend_func_extended"), glewGetExtension("GL_EXT_blend_func_extended"));

  glewInfoFunc(fi, "glBindFragDataLocationIndexedEXT", glBindFragDataLocationIndexedEXT == NULL);
  glewInfoFunc(fi, "glGetFragDataIndexEXT", glGetFragDataIndexEXT == NULL);
  glewInfoFunc(fi, "glGetProgramResourceLocationIndexEXT", glGetProgramResourceLocationIndexEXT == NULL);
}

#endif /* GL_EXT_blend_func_extended */

#ifdef GL_EXT_blend_func_separate

static void _glewInfo_GL_EXT_blend_func_separate (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_blend_func_separate", GLEW_EXT_blend_func_separate, glewIsSupported("GL_EXT_blend_func_separate"), glewGetExtension("GL_EXT_blend_func_separate"));

  glewInfoFunc(fi, "glBlendFuncSeparateEXT", glBlendFuncSeparateEXT == NULL);
}

#endif /* GL_EXT_blend_func_separate */

#ifdef GL_EXT_blend_logic_op

static void _glewInfo_GL_EXT_blend_logic_op (void)
{
  glewPrintExt("GL_EXT_blend_logic_op", GLEW_EXT_blend_logic_op, glewIsSupported("GL_EXT_blend_logic_op"), glewGetExtension("GL_EXT_blend_logic_op"));
}

#endif /* GL_EXT_blend_logic_op */

#ifdef GL_EXT_blend_minmax

static void _glewInfo_GL_EXT_blend_minmax (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_blend_minmax", GLEW_EXT_blend_minmax, glewIsSupported("GL_EXT_blend_minmax"), glewGetExtension("GL_EXT_blend_minmax"));

  glewInfoFunc(fi, "glBlendEquationEXT", glBlendEquationEXT == NULL);
}

#endif /* GL_EXT_blend_minmax */

#ifdef GL_EXT_blend_subtract

static void _glewInfo_GL_EXT_blend_subtract (void)
{
  glewPrintExt("GL_EXT_blend_subtract", GLEW_EXT_blend_subtract, glewIsSupported("GL_EXT_blend_subtract"), glewGetExtension("GL_EXT_blend_subtract"));
}

#endif /* GL_EXT_blend_subtract */

#ifdef GL_EXT_buffer_storage

static void _glewInfo_GL_EXT_buffer_storage (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_buffer_storage", GLEW_EXT_buffer_storage, glewIsSupported("GL_EXT_buffer_storage"), glewGetExtension("GL_EXT_buffer_storage"));

  glewInfoFunc(fi, "glBufferStorageEXT", glBufferStorageEXT == NULL);
  glewInfoFunc(fi, "glNamedBufferStorageEXT", glNamedBufferStorageEXT == NULL);
}

#endif /* GL_EXT_buffer_storage */

#ifdef GL_EXT_clear_texture

static void _glewInfo_GL_EXT_clear_texture (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_clear_texture", GLEW_EXT_clear_texture, glewIsSupported("GL_EXT_clear_texture"), glewGetExtension("GL_EXT_clear_texture"));

  glewInfoFunc(fi, "glClearTexImageEXT", glClearTexImageEXT == NULL);
  glewInfoFunc(fi, "glClearTexSubImageEXT", glClearTexSubImageEXT == NULL);
}

#endif /* GL_EXT_clear_texture */

#ifdef GL_EXT_clip_control

static void _glewInfo_GL_EXT_clip_control (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_clip_control", GLEW_EXT_clip_control, glewIsSupported("GL_EXT_clip_control"), glewGetExtension("GL_EXT_clip_control"));

  glewInfoFunc(fi, "glClipControlEXT", glClipControlEXT == NULL);
}

#endif /* GL_EXT_clip_control */

#ifdef GL_EXT_clip_cull_distance

static void _glewInfo_GL_EXT_clip_cull_distance (void)
{
  glewPrintExt("GL_EXT_clip_cull_distance", GLEW_EXT_clip_cull_distance, glewIsSupported("GL_EXT_clip_cull_distance"), glewGetExtension("GL_EXT_clip_cull_distance"));
}

#endif /* GL_EXT_clip_cull_distance */

#ifdef GL_EXT_clip_volume_hint

static void _glewInfo_GL_EXT_clip_volume_hint (void)
{
  glewPrintExt("GL_EXT_clip_volume_hint", GLEW_EXT_clip_volume_hint, glewIsSupported("GL_EXT_clip_volume_hint"), glewGetExtension("GL_EXT_clip_volume_hint"));
}

#endif /* GL_EXT_clip_volume_hint */

#ifdef GL_EXT_cmyka

static void _glewInfo_GL_EXT_cmyka (void)
{
  glewPrintExt("GL_EXT_cmyka", GLEW_EXT_cmyka, glewIsSupported("GL_EXT_cmyka"), glewGetExtension("GL_EXT_cmyka"));
}

#endif /* GL_EXT_cmyka */

#ifdef GL_EXT_color_buffer_float

static void _glewInfo_GL_EXT_color_buffer_float (void)
{
  glewPrintExt("GL_EXT_color_buffer_float", GLEW_EXT_color_buffer_float, glewIsSupported("GL_EXT_color_buffer_float"), glewGetExtension("GL_EXT_color_buffer_float"));
}

#endif /* GL_EXT_color_buffer_float */

#ifdef GL_EXT_color_buffer_half_float

static void _glewInfo_GL_EXT_color_buffer_half_float (void)
{
  glewPrintExt("GL_EXT_color_buffer_half_float", GLEW_EXT_color_buffer_half_float, glewIsSupported("GL_EXT_color_buffer_half_float"), glewGetExtension("GL_EXT_color_buffer_half_float"));
}

#endif /* GL_EXT_color_buffer_half_float */

#ifdef GL_EXT_color_subtable

static void _glewInfo_GL_EXT_color_subtable (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_color_subtable", GLEW_EXT_color_subtable, glewIsSupported("GL_EXT_color_subtable"), glewGetExtension("GL_EXT_color_subtable"));

  glewInfoFunc(fi, "glColorSubTableEXT", glColorSubTableEXT == NULL);
  glewInfoFunc(fi, "glCopyColorSubTableEXT", glCopyColorSubTableEXT == NULL);
}

#endif /* GL_EXT_color_subtable */

#ifdef GL_EXT_compiled_vertex_array

static void _glewInfo_GL_EXT_compiled_vertex_array (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_compiled_vertex_array", GLEW_EXT_compiled_vertex_array, glewIsSupported("GL_EXT_compiled_vertex_array"), glewGetExtension("GL_EXT_compiled_vertex_array"));

  glewInfoFunc(fi, "glLockArraysEXT", glLockArraysEXT == NULL);
  glewInfoFunc(fi, "glUnlockArraysEXT", glUnlockArraysEXT == NULL);
}

#endif /* GL_EXT_compiled_vertex_array */

#ifdef GL_EXT_compressed_ETC1_RGB8_sub_texture

static void _glewInfo_GL_EXT_compressed_ETC1_RGB8_sub_texture (void)
{
  glewPrintExt("GL_EXT_compressed_ETC1_RGB8_sub_texture", GLEW_EXT_compressed_ETC1_RGB8_sub_texture, glewIsSupported("GL_EXT_compressed_ETC1_RGB8_sub_texture"), glewGetExtension("GL_EXT_compressed_ETC1_RGB8_sub_texture"));
}

#endif /* GL_EXT_compressed_ETC1_RGB8_sub_texture */

#ifdef GL_EXT_conservative_depth

static void _glewInfo_GL_EXT_conservative_depth (void)
{
  glewPrintExt("GL_EXT_conservative_depth", GLEW_EXT_conservative_depth, glewIsSupported("GL_EXT_conservative_depth"), glewGetExtension("GL_EXT_conservative_depth"));
}

#endif /* GL_EXT_conservative_depth */

#ifdef GL_EXT_convolution

static void _glewInfo_GL_EXT_convolution (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_convolution", GLEW_EXT_convolution, glewIsSupported("GL_EXT_convolution"), glewGetExtension("GL_EXT_convolution"));

  glewInfoFunc(fi, "glConvolutionFilter1DEXT", glConvolutionFilter1DEXT == NULL);
  glewInfoFunc(fi, "glConvolutionFilter2DEXT", glConvolutionFilter2DEXT == NULL);
  glewInfoFunc(fi, "glConvolutionParameterfEXT", glConvolutionParameterfEXT == NULL);
  glewInfoFunc(fi, "glConvolutionParameterfvEXT", glConvolutionParameterfvEXT == NULL);
  glewInfoFunc(fi, "glConvolutionParameteriEXT", glConvolutionParameteriEXT == NULL);
  glewInfoFunc(fi, "glConvolutionParameterivEXT", glConvolutionParameterivEXT == NULL);
  glewInfoFunc(fi, "glCopyConvolutionFilter1DEXT", glCopyConvolutionFilter1DEXT == NULL);
  glewInfoFunc(fi, "glCopyConvolutionFilter2DEXT", glCopyConvolutionFilter2DEXT == NULL);
  glewInfoFunc(fi, "glGetConvolutionFilterEXT", glGetConvolutionFilterEXT == NULL);
  glewInfoFunc(fi, "glGetConvolutionParameterfvEXT", glGetConvolutionParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetConvolutionParameterivEXT", glGetConvolutionParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetSeparableFilterEXT", glGetSeparableFilterEXT == NULL);
  glewInfoFunc(fi, "glSeparableFilter2DEXT", glSeparableFilter2DEXT == NULL);
}

#endif /* GL_EXT_convolution */

#ifdef GL_EXT_coordinate_frame

static void _glewInfo_GL_EXT_coordinate_frame (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_coordinate_frame", GLEW_EXT_coordinate_frame, glewIsSupported("GL_EXT_coordinate_frame"), glewGetExtension("GL_EXT_coordinate_frame"));

  glewInfoFunc(fi, "glBinormalPointerEXT", glBinormalPointerEXT == NULL);
  glewInfoFunc(fi, "glTangentPointerEXT", glTangentPointerEXT == NULL);
}

#endif /* GL_EXT_coordinate_frame */

#ifdef GL_EXT_copy_image

static void _glewInfo_GL_EXT_copy_image (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_copy_image", GLEW_EXT_copy_image, glewIsSupported("GL_EXT_copy_image"), glewGetExtension("GL_EXT_copy_image"));

  glewInfoFunc(fi, "glCopyImageSubDataEXT", glCopyImageSubDataEXT == NULL);
}

#endif /* GL_EXT_copy_image */

#ifdef GL_EXT_copy_texture

static void _glewInfo_GL_EXT_copy_texture (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_copy_texture", GLEW_EXT_copy_texture, glewIsSupported("GL_EXT_copy_texture"), glewGetExtension("GL_EXT_copy_texture"));

  glewInfoFunc(fi, "glCopyTexImage1DEXT", glCopyTexImage1DEXT == NULL);
  glewInfoFunc(fi, "glCopyTexImage2DEXT", glCopyTexImage2DEXT == NULL);
  glewInfoFunc(fi, "glCopyTexSubImage1DEXT", glCopyTexSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glCopyTexSubImage2DEXT", glCopyTexSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glCopyTexSubImage3DEXT", glCopyTexSubImage3DEXT == NULL);
}

#endif /* GL_EXT_copy_texture */

#ifdef GL_EXT_cull_vertex

static void _glewInfo_GL_EXT_cull_vertex (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_cull_vertex", GLEW_EXT_cull_vertex, glewIsSupported("GL_EXT_cull_vertex"), glewGetExtension("GL_EXT_cull_vertex"));

  glewInfoFunc(fi, "glCullParameterdvEXT", glCullParameterdvEXT == NULL);
  glewInfoFunc(fi, "glCullParameterfvEXT", glCullParameterfvEXT == NULL);
}

#endif /* GL_EXT_cull_vertex */

#ifdef GL_EXT_debug_label

static void _glewInfo_GL_EXT_debug_label (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_debug_label", GLEW_EXT_debug_label, glewIsSupported("GL_EXT_debug_label"), glewGetExtension("GL_EXT_debug_label"));

  glewInfoFunc(fi, "glGetObjectLabelEXT", glGetObjectLabelEXT == NULL);
  glewInfoFunc(fi, "glLabelObjectEXT", glLabelObjectEXT == NULL);
}

#endif /* GL_EXT_debug_label */

#ifdef GL_EXT_debug_marker

static void _glewInfo_GL_EXT_debug_marker (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_debug_marker", GLEW_EXT_debug_marker, glewIsSupported("GL_EXT_debug_marker"), glewGetExtension("GL_EXT_debug_marker"));

  glewInfoFunc(fi, "glInsertEventMarkerEXT", glInsertEventMarkerEXT == NULL);
  glewInfoFunc(fi, "glPopGroupMarkerEXT", glPopGroupMarkerEXT == NULL);
  glewInfoFunc(fi, "glPushGroupMarkerEXT", glPushGroupMarkerEXT == NULL);
}

#endif /* GL_EXT_debug_marker */

#ifdef GL_EXT_depth_bounds_test

static void _glewInfo_GL_EXT_depth_bounds_test (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_depth_bounds_test", GLEW_EXT_depth_bounds_test, glewIsSupported("GL_EXT_depth_bounds_test"), glewGetExtension("GL_EXT_depth_bounds_test"));

  glewInfoFunc(fi, "glDepthBoundsEXT", glDepthBoundsEXT == NULL);
}

#endif /* GL_EXT_depth_bounds_test */

#ifdef GL_EXT_depth_clamp

static void _glewInfo_GL_EXT_depth_clamp (void)
{
  glewPrintExt("GL_EXT_depth_clamp", GLEW_EXT_depth_clamp, glewIsSupported("GL_EXT_depth_clamp"), glewGetExtension("GL_EXT_depth_clamp"));
}

#endif /* GL_EXT_depth_clamp */

#ifdef GL_EXT_direct_state_access

static void _glewInfo_GL_EXT_direct_state_access (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_direct_state_access", GLEW_EXT_direct_state_access, glewIsSupported("GL_EXT_direct_state_access"), glewGetExtension("GL_EXT_direct_state_access"));

  glewInfoFunc(fi, "glBindMultiTextureEXT", glBindMultiTextureEXT == NULL);
  glewInfoFunc(fi, "glCheckNamedFramebufferStatusEXT", glCheckNamedFramebufferStatusEXT == NULL);
  glewInfoFunc(fi, "glClientAttribDefaultEXT", glClientAttribDefaultEXT == NULL);
  glewInfoFunc(fi, "glCompressedMultiTexImage1DEXT", glCompressedMultiTexImage1DEXT == NULL);
  glewInfoFunc(fi, "glCompressedMultiTexImage2DEXT", glCompressedMultiTexImage2DEXT == NULL);
  glewInfoFunc(fi, "glCompressedMultiTexImage3DEXT", glCompressedMultiTexImage3DEXT == NULL);
  glewInfoFunc(fi, "glCompressedMultiTexSubImage1DEXT", glCompressedMultiTexSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glCompressedMultiTexSubImage2DEXT", glCompressedMultiTexSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glCompressedMultiTexSubImage3DEXT", glCompressedMultiTexSubImage3DEXT == NULL);
  glewInfoFunc(fi, "glCompressedTextureImage1DEXT", glCompressedTextureImage1DEXT == NULL);
  glewInfoFunc(fi, "glCompressedTextureImage2DEXT", glCompressedTextureImage2DEXT == NULL);
  glewInfoFunc(fi, "glCompressedTextureImage3DEXT", glCompressedTextureImage3DEXT == NULL);
  glewInfoFunc(fi, "glCompressedTextureSubImage1DEXT", glCompressedTextureSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glCompressedTextureSubImage2DEXT", glCompressedTextureSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glCompressedTextureSubImage3DEXT", glCompressedTextureSubImage3DEXT == NULL);
  glewInfoFunc(fi, "glCopyMultiTexImage1DEXT", glCopyMultiTexImage1DEXT == NULL);
  glewInfoFunc(fi, "glCopyMultiTexImage2DEXT", glCopyMultiTexImage2DEXT == NULL);
  glewInfoFunc(fi, "glCopyMultiTexSubImage1DEXT", glCopyMultiTexSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glCopyMultiTexSubImage2DEXT", glCopyMultiTexSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glCopyMultiTexSubImage3DEXT", glCopyMultiTexSubImage3DEXT == NULL);
  glewInfoFunc(fi, "glCopyTextureImage1DEXT", glCopyTextureImage1DEXT == NULL);
  glewInfoFunc(fi, "glCopyTextureImage2DEXT", glCopyTextureImage2DEXT == NULL);
  glewInfoFunc(fi, "glCopyTextureSubImage1DEXT", glCopyTextureSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glCopyTextureSubImage2DEXT", glCopyTextureSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glCopyTextureSubImage3DEXT", glCopyTextureSubImage3DEXT == NULL);
  glewInfoFunc(fi, "glDisableClientStateIndexedEXT", glDisableClientStateIndexedEXT == NULL);
  glewInfoFunc(fi, "glDisableClientStateiEXT", glDisableClientStateiEXT == NULL);
  glewInfoFunc(fi, "glDisableVertexArrayAttribEXT", glDisableVertexArrayAttribEXT == NULL);
  glewInfoFunc(fi, "glDisableVertexArrayEXT", glDisableVertexArrayEXT == NULL);
  glewInfoFunc(fi, "glEnableClientStateIndexedEXT", glEnableClientStateIndexedEXT == NULL);
  glewInfoFunc(fi, "glEnableClientStateiEXT", glEnableClientStateiEXT == NULL);
  glewInfoFunc(fi, "glEnableVertexArrayAttribEXT", glEnableVertexArrayAttribEXT == NULL);
  glewInfoFunc(fi, "glEnableVertexArrayEXT", glEnableVertexArrayEXT == NULL);
  glewInfoFunc(fi, "glFlushMappedNamedBufferRangeEXT", glFlushMappedNamedBufferRangeEXT == NULL);
  glewInfoFunc(fi, "glFramebufferDrawBufferEXT", glFramebufferDrawBufferEXT == NULL);
  glewInfoFunc(fi, "glFramebufferDrawBuffersEXT", glFramebufferDrawBuffersEXT == NULL);
  glewInfoFunc(fi, "glFramebufferReadBufferEXT", glFramebufferReadBufferEXT == NULL);
  glewInfoFunc(fi, "glGenerateMultiTexMipmapEXT", glGenerateMultiTexMipmapEXT == NULL);
  glewInfoFunc(fi, "glGenerateTextureMipmapEXT", glGenerateTextureMipmapEXT == NULL);
  glewInfoFunc(fi, "glGetCompressedMultiTexImageEXT", glGetCompressedMultiTexImageEXT == NULL);
  glewInfoFunc(fi, "glGetCompressedTextureImageEXT", glGetCompressedTextureImageEXT == NULL);
  glewInfoFunc(fi, "glGetDoubleIndexedvEXT", glGetDoubleIndexedvEXT == NULL);
  glewInfoFunc(fi, "glGetDoublei_vEXT", glGetDoublei_vEXT == NULL);
  glewInfoFunc(fi, "glGetFloatIndexedvEXT", glGetFloatIndexedvEXT == NULL);
  glewInfoFunc(fi, "glGetFloati_vEXT", glGetFloati_vEXT == NULL);
  glewInfoFunc(fi, "glGetFramebufferParameterivEXT", glGetFramebufferParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexEnvfvEXT", glGetMultiTexEnvfvEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexEnvivEXT", glGetMultiTexEnvivEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexGendvEXT", glGetMultiTexGendvEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexGenfvEXT", glGetMultiTexGenfvEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexGenivEXT", glGetMultiTexGenivEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexImageEXT", glGetMultiTexImageEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexLevelParameterfvEXT", glGetMultiTexLevelParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexLevelParameterivEXT", glGetMultiTexLevelParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexParameterIivEXT", glGetMultiTexParameterIivEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexParameterIuivEXT", glGetMultiTexParameterIuivEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexParameterfvEXT", glGetMultiTexParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetMultiTexParameterivEXT", glGetMultiTexParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetNamedBufferParameterivEXT", glGetNamedBufferParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetNamedBufferPointervEXT", glGetNamedBufferPointervEXT == NULL);
  glewInfoFunc(fi, "glGetNamedBufferSubDataEXT", glGetNamedBufferSubDataEXT == NULL);
  glewInfoFunc(fi, "glGetNamedFramebufferAttachmentParameterivEXT", glGetNamedFramebufferAttachmentParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetNamedProgramLocalParameterIivEXT", glGetNamedProgramLocalParameterIivEXT == NULL);
  glewInfoFunc(fi, "glGetNamedProgramLocalParameterIuivEXT", glGetNamedProgramLocalParameterIuivEXT == NULL);
  glewInfoFunc(fi, "glGetNamedProgramLocalParameterdvEXT", glGetNamedProgramLocalParameterdvEXT == NULL);
  glewInfoFunc(fi, "glGetNamedProgramLocalParameterfvEXT", glGetNamedProgramLocalParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetNamedProgramStringEXT", glGetNamedProgramStringEXT == NULL);
  glewInfoFunc(fi, "glGetNamedProgramivEXT", glGetNamedProgramivEXT == NULL);
  glewInfoFunc(fi, "glGetNamedRenderbufferParameterivEXT", glGetNamedRenderbufferParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetPointerIndexedvEXT", glGetPointerIndexedvEXT == NULL);
  glewInfoFunc(fi, "glGetPointeri_vEXT", glGetPointeri_vEXT == NULL);
  glewInfoFunc(fi, "glGetTextureImageEXT", glGetTextureImageEXT == NULL);
  glewInfoFunc(fi, "glGetTextureLevelParameterfvEXT", glGetTextureLevelParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetTextureLevelParameterivEXT", glGetTextureLevelParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetTextureParameterIivEXT", glGetTextureParameterIivEXT == NULL);
  glewInfoFunc(fi, "glGetTextureParameterIuivEXT", glGetTextureParameterIuivEXT == NULL);
  glewInfoFunc(fi, "glGetTextureParameterfvEXT", glGetTextureParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetTextureParameterivEXT", glGetTextureParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetVertexArrayIntegeri_vEXT", glGetVertexArrayIntegeri_vEXT == NULL);
  glewInfoFunc(fi, "glGetVertexArrayIntegervEXT", glGetVertexArrayIntegervEXT == NULL);
  glewInfoFunc(fi, "glGetVertexArrayPointeri_vEXT", glGetVertexArrayPointeri_vEXT == NULL);
  glewInfoFunc(fi, "glGetVertexArrayPointervEXT", glGetVertexArrayPointervEXT == NULL);
  glewInfoFunc(fi, "glMapNamedBufferEXT", glMapNamedBufferEXT == NULL);
  glewInfoFunc(fi, "glMapNamedBufferRangeEXT", glMapNamedBufferRangeEXT == NULL);
  glewInfoFunc(fi, "glMatrixFrustumEXT", glMatrixFrustumEXT == NULL);
  glewInfoFunc(fi, "glMatrixLoadIdentityEXT", glMatrixLoadIdentityEXT == NULL);
  glewInfoFunc(fi, "glMatrixLoadTransposedEXT", glMatrixLoadTransposedEXT == NULL);
  glewInfoFunc(fi, "glMatrixLoadTransposefEXT", glMatrixLoadTransposefEXT == NULL);
  glewInfoFunc(fi, "glMatrixLoaddEXT", glMatrixLoaddEXT == NULL);
  glewInfoFunc(fi, "glMatrixLoadfEXT", glMatrixLoadfEXT == NULL);
  glewInfoFunc(fi, "glMatrixMultTransposedEXT", glMatrixMultTransposedEXT == NULL);
  glewInfoFunc(fi, "glMatrixMultTransposefEXT", glMatrixMultTransposefEXT == NULL);
  glewInfoFunc(fi, "glMatrixMultdEXT", glMatrixMultdEXT == NULL);
  glewInfoFunc(fi, "glMatrixMultfEXT", glMatrixMultfEXT == NULL);
  glewInfoFunc(fi, "glMatrixOrthoEXT", glMatrixOrthoEXT == NULL);
  glewInfoFunc(fi, "glMatrixPopEXT", glMatrixPopEXT == NULL);
  glewInfoFunc(fi, "glMatrixPushEXT", glMatrixPushEXT == NULL);
  glewInfoFunc(fi, "glMatrixRotatedEXT", glMatrixRotatedEXT == NULL);
  glewInfoFunc(fi, "glMatrixRotatefEXT", glMatrixRotatefEXT == NULL);
  glewInfoFunc(fi, "glMatrixScaledEXT", glMatrixScaledEXT == NULL);
  glewInfoFunc(fi, "glMatrixScalefEXT", glMatrixScalefEXT == NULL);
  glewInfoFunc(fi, "glMatrixTranslatedEXT", glMatrixTranslatedEXT == NULL);
  glewInfoFunc(fi, "glMatrixTranslatefEXT", glMatrixTranslatefEXT == NULL);
  glewInfoFunc(fi, "glMultiTexBufferEXT", glMultiTexBufferEXT == NULL);
  glewInfoFunc(fi, "glMultiTexCoordPointerEXT", glMultiTexCoordPointerEXT == NULL);
  glewInfoFunc(fi, "glMultiTexEnvfEXT", glMultiTexEnvfEXT == NULL);
  glewInfoFunc(fi, "glMultiTexEnvfvEXT", glMultiTexEnvfvEXT == NULL);
  glewInfoFunc(fi, "glMultiTexEnviEXT", glMultiTexEnviEXT == NULL);
  glewInfoFunc(fi, "glMultiTexEnvivEXT", glMultiTexEnvivEXT == NULL);
  glewInfoFunc(fi, "glMultiTexGendEXT", glMultiTexGendEXT == NULL);
  glewInfoFunc(fi, "glMultiTexGendvEXT", glMultiTexGendvEXT == NULL);
  glewInfoFunc(fi, "glMultiTexGenfEXT", glMultiTexGenfEXT == NULL);
  glewInfoFunc(fi, "glMultiTexGenfvEXT", glMultiTexGenfvEXT == NULL);
  glewInfoFunc(fi, "glMultiTexGeniEXT", glMultiTexGeniEXT == NULL);
  glewInfoFunc(fi, "glMultiTexGenivEXT", glMultiTexGenivEXT == NULL);
  glewInfoFunc(fi, "glMultiTexImage1DEXT", glMultiTexImage1DEXT == NULL);
  glewInfoFunc(fi, "glMultiTexImage2DEXT", glMultiTexImage2DEXT == NULL);
  glewInfoFunc(fi, "glMultiTexImage3DEXT", glMultiTexImage3DEXT == NULL);
  glewInfoFunc(fi, "glMultiTexParameterIivEXT", glMultiTexParameterIivEXT == NULL);
  glewInfoFunc(fi, "glMultiTexParameterIuivEXT", glMultiTexParameterIuivEXT == NULL);
  glewInfoFunc(fi, "glMultiTexParameterfEXT", glMultiTexParameterfEXT == NULL);
  glewInfoFunc(fi, "glMultiTexParameterfvEXT", glMultiTexParameterfvEXT == NULL);
  glewInfoFunc(fi, "glMultiTexParameteriEXT", glMultiTexParameteriEXT == NULL);
  glewInfoFunc(fi, "glMultiTexParameterivEXT", glMultiTexParameterivEXT == NULL);
  glewInfoFunc(fi, "glMultiTexRenderbufferEXT", glMultiTexRenderbufferEXT == NULL);
  glewInfoFunc(fi, "glMultiTexSubImage1DEXT", glMultiTexSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glMultiTexSubImage2DEXT", glMultiTexSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glMultiTexSubImage3DEXT", glMultiTexSubImage3DEXT == NULL);
  glewInfoFunc(fi, "glNamedBufferDataEXT", glNamedBufferDataEXT == NULL);
  glewInfoFunc(fi, "glNamedBufferSubDataEXT", glNamedBufferSubDataEXT == NULL);
  glewInfoFunc(fi, "glNamedCopyBufferSubDataEXT", glNamedCopyBufferSubDataEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferRenderbufferEXT", glNamedFramebufferRenderbufferEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTexture1DEXT", glNamedFramebufferTexture1DEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTexture2DEXT", glNamedFramebufferTexture2DEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTexture3DEXT", glNamedFramebufferTexture3DEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTextureEXT", glNamedFramebufferTextureEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTextureFaceEXT", glNamedFramebufferTextureFaceEXT == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTextureLayerEXT", glNamedFramebufferTextureLayerEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameter4dEXT", glNamedProgramLocalParameter4dEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameter4dvEXT", glNamedProgramLocalParameter4dvEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameter4fEXT", glNamedProgramLocalParameter4fEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameter4fvEXT", glNamedProgramLocalParameter4fvEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameterI4iEXT", glNamedProgramLocalParameterI4iEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameterI4ivEXT", glNamedProgramLocalParameterI4ivEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameterI4uiEXT", glNamedProgramLocalParameterI4uiEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameterI4uivEXT", glNamedProgramLocalParameterI4uivEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParameters4fvEXT", glNamedProgramLocalParameters4fvEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParametersI4ivEXT", glNamedProgramLocalParametersI4ivEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramLocalParametersI4uivEXT", glNamedProgramLocalParametersI4uivEXT == NULL);
  glewInfoFunc(fi, "glNamedProgramStringEXT", glNamedProgramStringEXT == NULL);
  glewInfoFunc(fi, "glNamedRenderbufferStorageEXT", glNamedRenderbufferStorageEXT == NULL);
  glewInfoFunc(fi, "glNamedRenderbufferStorageMultisampleCoverageEXT", glNamedRenderbufferStorageMultisampleCoverageEXT == NULL);
  glewInfoFunc(fi, "glNamedRenderbufferStorageMultisampleEXT", glNamedRenderbufferStorageMultisampleEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform1fEXT", glProgramUniform1fEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform1fvEXT", glProgramUniform1fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform1iEXT", glProgramUniform1iEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform1ivEXT", glProgramUniform1ivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform1uiEXT", glProgramUniform1uiEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform1uivEXT", glProgramUniform1uivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform2fEXT", glProgramUniform2fEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform2fvEXT", glProgramUniform2fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform2iEXT", glProgramUniform2iEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform2ivEXT", glProgramUniform2ivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform2uiEXT", glProgramUniform2uiEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform2uivEXT", glProgramUniform2uivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform3fEXT", glProgramUniform3fEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform3fvEXT", glProgramUniform3fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform3iEXT", glProgramUniform3iEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform3ivEXT", glProgramUniform3ivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform3uiEXT", glProgramUniform3uiEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform3uivEXT", glProgramUniform3uivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform4fEXT", glProgramUniform4fEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform4fvEXT", glProgramUniform4fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform4iEXT", glProgramUniform4iEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform4ivEXT", glProgramUniform4ivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform4uiEXT", glProgramUniform4uiEXT == NULL);
  glewInfoFunc(fi, "glProgramUniform4uivEXT", glProgramUniform4uivEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2fvEXT", glProgramUniformMatrix2fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2x3fvEXT", glProgramUniformMatrix2x3fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix2x4fvEXT", glProgramUniformMatrix2x4fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3fvEXT", glProgramUniformMatrix3fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3x2fvEXT", glProgramUniformMatrix3x2fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix3x4fvEXT", glProgramUniformMatrix3x4fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4fvEXT", glProgramUniformMatrix4fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4x2fvEXT", glProgramUniformMatrix4x2fvEXT == NULL);
  glewInfoFunc(fi, "glProgramUniformMatrix4x3fvEXT", glProgramUniformMatrix4x3fvEXT == NULL);
  glewInfoFunc(fi, "glPushClientAttribDefaultEXT", glPushClientAttribDefaultEXT == NULL);
  glewInfoFunc(fi, "glTextureBufferEXT", glTextureBufferEXT == NULL);
  glewInfoFunc(fi, "glTextureImage1DEXT", glTextureImage1DEXT == NULL);
  glewInfoFunc(fi, "glTextureImage2DEXT", glTextureImage2DEXT == NULL);
  glewInfoFunc(fi, "glTextureImage3DEXT", glTextureImage3DEXT == NULL);
  glewInfoFunc(fi, "glTextureParameterIivEXT", glTextureParameterIivEXT == NULL);
  glewInfoFunc(fi, "glTextureParameterIuivEXT", glTextureParameterIuivEXT == NULL);
  glewInfoFunc(fi, "glTextureParameterfEXT", glTextureParameterfEXT == NULL);
  glewInfoFunc(fi, "glTextureParameterfvEXT", glTextureParameterfvEXT == NULL);
  glewInfoFunc(fi, "glTextureParameteriEXT", glTextureParameteriEXT == NULL);
  glewInfoFunc(fi, "glTextureParameterivEXT", glTextureParameterivEXT == NULL);
  glewInfoFunc(fi, "glTextureRenderbufferEXT", glTextureRenderbufferEXT == NULL);
  glewInfoFunc(fi, "glTextureSubImage1DEXT", glTextureSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glTextureSubImage2DEXT", glTextureSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glTextureSubImage3DEXT", glTextureSubImage3DEXT == NULL);
  glewInfoFunc(fi, "glUnmapNamedBufferEXT", glUnmapNamedBufferEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayColorOffsetEXT", glVertexArrayColorOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayEdgeFlagOffsetEXT", glVertexArrayEdgeFlagOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayFogCoordOffsetEXT", glVertexArrayFogCoordOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayIndexOffsetEXT", glVertexArrayIndexOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayMultiTexCoordOffsetEXT", glVertexArrayMultiTexCoordOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayNormalOffsetEXT", glVertexArrayNormalOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArraySecondaryColorOffsetEXT", glVertexArraySecondaryColorOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayTexCoordOffsetEXT", glVertexArrayTexCoordOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribDivisorEXT", glVertexArrayVertexAttribDivisorEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribIOffsetEXT", glVertexArrayVertexAttribIOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribOffsetEXT", glVertexArrayVertexAttribOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexOffsetEXT", glVertexArrayVertexOffsetEXT == NULL);
}

#endif /* GL_EXT_direct_state_access */

#ifdef GL_EXT_discard_framebuffer

static void _glewInfo_GL_EXT_discard_framebuffer (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_discard_framebuffer", GLEW_EXT_discard_framebuffer, glewIsSupported("GL_EXT_discard_framebuffer"), glewGetExtension("GL_EXT_discard_framebuffer"));

  glewInfoFunc(fi, "glDiscardFramebufferEXT", glDiscardFramebufferEXT == NULL);
}

#endif /* GL_EXT_discard_framebuffer */

#ifdef GL_EXT_disjoint_timer_query

static void _glewInfo_GL_EXT_disjoint_timer_query (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_disjoint_timer_query", GLEW_EXT_disjoint_timer_query, glewIsSupported("GL_EXT_disjoint_timer_query"), glewGetExtension("GL_EXT_disjoint_timer_query"));

  glewInfoFunc(fi, "glBeginQueryEXT", glBeginQueryEXT == NULL);
  glewInfoFunc(fi, "glDeleteQueriesEXT", glDeleteQueriesEXT == NULL);
  glewInfoFunc(fi, "glEndQueryEXT", glEndQueryEXT == NULL);
  glewInfoFunc(fi, "glGenQueriesEXT", glGenQueriesEXT == NULL);
  glewInfoFunc(fi, "glGetInteger64vEXT", glGetInteger64vEXT == NULL);
  glewInfoFunc(fi, "glGetQueryObjectivEXT", glGetQueryObjectivEXT == NULL);
  glewInfoFunc(fi, "glGetQueryObjectuivEXT", glGetQueryObjectuivEXT == NULL);
  glewInfoFunc(fi, "glGetQueryivEXT", glGetQueryivEXT == NULL);
  glewInfoFunc(fi, "glIsQueryEXT", glIsQueryEXT == NULL);
  glewInfoFunc(fi, "glQueryCounterEXT", glQueryCounterEXT == NULL);
}

#endif /* GL_EXT_disjoint_timer_query */

#ifdef GL_EXT_draw_buffers

static void _glewInfo_GL_EXT_draw_buffers (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_draw_buffers", GLEW_EXT_draw_buffers, glewIsSupported("GL_EXT_draw_buffers"), glewGetExtension("GL_EXT_draw_buffers"));

  glewInfoFunc(fi, "glDrawBuffersEXT", glDrawBuffersEXT == NULL);
}

#endif /* GL_EXT_draw_buffers */

#ifdef GL_EXT_draw_buffers2

static void _glewInfo_GL_EXT_draw_buffers2 (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_draw_buffers2", GLEW_EXT_draw_buffers2, glewIsSupported("GL_EXT_draw_buffers2"), glewGetExtension("GL_EXT_draw_buffers2"));

  glewInfoFunc(fi, "glColorMaskIndexedEXT", glColorMaskIndexedEXT == NULL);
  glewInfoFunc(fi, "glDisableIndexedEXT", glDisableIndexedEXT == NULL);
  glewInfoFunc(fi, "glEnableIndexedEXT", glEnableIndexedEXT == NULL);
  glewInfoFunc(fi, "glGetBooleanIndexedvEXT", glGetBooleanIndexedvEXT == NULL);
  glewInfoFunc(fi, "glGetIntegerIndexedvEXT", glGetIntegerIndexedvEXT == NULL);
  glewInfoFunc(fi, "glIsEnabledIndexedEXT", glIsEnabledIndexedEXT == NULL);
}

#endif /* GL_EXT_draw_buffers2 */

#ifdef GL_EXT_draw_buffers_indexed

static void _glewInfo_GL_EXT_draw_buffers_indexed (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_draw_buffers_indexed", GLEW_EXT_draw_buffers_indexed, glewIsSupported("GL_EXT_draw_buffers_indexed"), glewGetExtension("GL_EXT_draw_buffers_indexed"));

  glewInfoFunc(fi, "glBlendEquationSeparateiEXT", glBlendEquationSeparateiEXT == NULL);
  glewInfoFunc(fi, "glBlendEquationiEXT", glBlendEquationiEXT == NULL);
  glewInfoFunc(fi, "glBlendFuncSeparateiEXT", glBlendFuncSeparateiEXT == NULL);
  glewInfoFunc(fi, "glBlendFunciEXT", glBlendFunciEXT == NULL);
  glewInfoFunc(fi, "glColorMaskiEXT", glColorMaskiEXT == NULL);
  glewInfoFunc(fi, "glDisableiEXT", glDisableiEXT == NULL);
  glewInfoFunc(fi, "glEnableiEXT", glEnableiEXT == NULL);
  glewInfoFunc(fi, "glIsEnablediEXT", glIsEnablediEXT == NULL);
}

#endif /* GL_EXT_draw_buffers_indexed */

#ifdef GL_EXT_draw_elements_base_vertex

static void _glewInfo_GL_EXT_draw_elements_base_vertex (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_draw_elements_base_vertex", GLEW_EXT_draw_elements_base_vertex, glewIsSupported("GL_EXT_draw_elements_base_vertex"), glewGetExtension("GL_EXT_draw_elements_base_vertex"));

  glewInfoFunc(fi, "glDrawElementsBaseVertexEXT", glDrawElementsBaseVertexEXT == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedBaseVertexEXT", glDrawElementsInstancedBaseVertexEXT == NULL);
  glewInfoFunc(fi, "glDrawRangeElementsBaseVertexEXT", glDrawRangeElementsBaseVertexEXT == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsBaseVertexEXT", glMultiDrawElementsBaseVertexEXT == NULL);
}

#endif /* GL_EXT_draw_elements_base_vertex */

#ifdef GL_EXT_draw_instanced

static void _glewInfo_GL_EXT_draw_instanced (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_draw_instanced", GLEW_EXT_draw_instanced, glewIsSupported("GL_EXT_draw_instanced"), glewGetExtension("GL_EXT_draw_instanced"));

  glewInfoFunc(fi, "glDrawArraysInstancedEXT", glDrawArraysInstancedEXT == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedEXT", glDrawElementsInstancedEXT == NULL);
}

#endif /* GL_EXT_draw_instanced */

#ifdef GL_EXT_draw_range_elements

static void _glewInfo_GL_EXT_draw_range_elements (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_draw_range_elements", GLEW_EXT_draw_range_elements, glewIsSupported("GL_EXT_draw_range_elements"), glewGetExtension("GL_EXT_draw_range_elements"));

  glewInfoFunc(fi, "glDrawRangeElementsEXT", glDrawRangeElementsEXT == NULL);
}

#endif /* GL_EXT_draw_range_elements */

#ifdef GL_EXT_draw_transform_feedback

static void _glewInfo_GL_EXT_draw_transform_feedback (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_draw_transform_feedback", GLEW_EXT_draw_transform_feedback, glewIsSupported("GL_EXT_draw_transform_feedback"), glewGetExtension("GL_EXT_draw_transform_feedback"));

  glewInfoFunc(fi, "glDrawTransformFeedbackEXT", glDrawTransformFeedbackEXT == NULL);
  glewInfoFunc(fi, "glDrawTransformFeedbackInstancedEXT", glDrawTransformFeedbackInstancedEXT == NULL);
}

#endif /* GL_EXT_draw_transform_feedback */

#ifdef GL_EXT_external_buffer

static void _glewInfo_GL_EXT_external_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_external_buffer", GLEW_EXT_external_buffer, glewIsSupported("GL_EXT_external_buffer"), glewGetExtension("GL_EXT_external_buffer"));

  glewInfoFunc(fi, "glBufferStorageExternalEXT", glBufferStorageExternalEXT == NULL);
  glewInfoFunc(fi, "glNamedBufferStorageExternalEXT", glNamedBufferStorageExternalEXT == NULL);
}

#endif /* GL_EXT_external_buffer */

#ifdef GL_EXT_float_blend

static void _glewInfo_GL_EXT_float_blend (void)
{
  glewPrintExt("GL_EXT_float_blend", GLEW_EXT_float_blend, glewIsSupported("GL_EXT_float_blend"), glewGetExtension("GL_EXT_float_blend"));
}

#endif /* GL_EXT_float_blend */

#ifdef GL_EXT_fog_coord

static void _glewInfo_GL_EXT_fog_coord (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_fog_coord", GLEW_EXT_fog_coord, glewIsSupported("GL_EXT_fog_coord"), glewGetExtension("GL_EXT_fog_coord"));

  glewInfoFunc(fi, "glFogCoordPointerEXT", glFogCoordPointerEXT == NULL);
  glewInfoFunc(fi, "glFogCoorddEXT", glFogCoorddEXT == NULL);
  glewInfoFunc(fi, "glFogCoorddvEXT", glFogCoorddvEXT == NULL);
  glewInfoFunc(fi, "glFogCoordfEXT", glFogCoordfEXT == NULL);
  glewInfoFunc(fi, "glFogCoordfvEXT", glFogCoordfvEXT == NULL);
}

#endif /* GL_EXT_fog_coord */

#ifdef GL_EXT_frag_depth

static void _glewInfo_GL_EXT_frag_depth (void)
{
  glewPrintExt("GL_EXT_frag_depth", GLEW_EXT_frag_depth, glewIsSupported("GL_EXT_frag_depth"), glewGetExtension("GL_EXT_frag_depth"));
}

#endif /* GL_EXT_frag_depth */

#ifdef GL_EXT_fragment_lighting

static void _glewInfo_GL_EXT_fragment_lighting (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_fragment_lighting", GLEW_EXT_fragment_lighting, glewIsSupported("GL_EXT_fragment_lighting"), glewGetExtension("GL_EXT_fragment_lighting"));

  glewInfoFunc(fi, "glFragmentColorMaterialEXT", glFragmentColorMaterialEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightModelfEXT", glFragmentLightModelfEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightModelfvEXT", glFragmentLightModelfvEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightModeliEXT", glFragmentLightModeliEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightModelivEXT", glFragmentLightModelivEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightfEXT", glFragmentLightfEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightfvEXT", glFragmentLightfvEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightiEXT", glFragmentLightiEXT == NULL);
  glewInfoFunc(fi, "glFragmentLightivEXT", glFragmentLightivEXT == NULL);
  glewInfoFunc(fi, "glFragmentMaterialfEXT", glFragmentMaterialfEXT == NULL);
  glewInfoFunc(fi, "glFragmentMaterialfvEXT", glFragmentMaterialfvEXT == NULL);
  glewInfoFunc(fi, "glFragmentMaterialiEXT", glFragmentMaterialiEXT == NULL);
  glewInfoFunc(fi, "glFragmentMaterialivEXT", glFragmentMaterialivEXT == NULL);
  glewInfoFunc(fi, "glGetFragmentLightfvEXT", glGetFragmentLightfvEXT == NULL);
  glewInfoFunc(fi, "glGetFragmentLightivEXT", glGetFragmentLightivEXT == NULL);
  glewInfoFunc(fi, "glGetFragmentMaterialfvEXT", glGetFragmentMaterialfvEXT == NULL);
  glewInfoFunc(fi, "glGetFragmentMaterialivEXT", glGetFragmentMaterialivEXT == NULL);
  glewInfoFunc(fi, "glLightEnviEXT", glLightEnviEXT == NULL);
}

#endif /* GL_EXT_fragment_lighting */

#ifdef GL_EXT_framebuffer_blit

static void _glewInfo_GL_EXT_framebuffer_blit (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_framebuffer_blit", GLEW_EXT_framebuffer_blit, glewIsSupported("GL_EXT_framebuffer_blit"), glewGetExtension("GL_EXT_framebuffer_blit"));

  glewInfoFunc(fi, "glBlitFramebufferEXT", glBlitFramebufferEXT == NULL);
}

#endif /* GL_EXT_framebuffer_blit */

#ifdef GL_EXT_framebuffer_multisample

static void _glewInfo_GL_EXT_framebuffer_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_framebuffer_multisample", GLEW_EXT_framebuffer_multisample, glewIsSupported("GL_EXT_framebuffer_multisample"), glewGetExtension("GL_EXT_framebuffer_multisample"));

  glewInfoFunc(fi, "glRenderbufferStorageMultisampleEXT", glRenderbufferStorageMultisampleEXT == NULL);
}

#endif /* GL_EXT_framebuffer_multisample */

#ifdef GL_EXT_framebuffer_multisample_blit_scaled

static void _glewInfo_GL_EXT_framebuffer_multisample_blit_scaled (void)
{
  glewPrintExt("GL_EXT_framebuffer_multisample_blit_scaled", GLEW_EXT_framebuffer_multisample_blit_scaled, glewIsSupported("GL_EXT_framebuffer_multisample_blit_scaled"), glewGetExtension("GL_EXT_framebuffer_multisample_blit_scaled"));
}

#endif /* GL_EXT_framebuffer_multisample_blit_scaled */

#ifdef GL_EXT_framebuffer_object

static void _glewInfo_GL_EXT_framebuffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_framebuffer_object", GLEW_EXT_framebuffer_object, glewIsSupported("GL_EXT_framebuffer_object"), glewGetExtension("GL_EXT_framebuffer_object"));

  glewInfoFunc(fi, "glBindFramebufferEXT", glBindFramebufferEXT == NULL);
  glewInfoFunc(fi, "glBindRenderbufferEXT", glBindRenderbufferEXT == NULL);
  glewInfoFunc(fi, "glCheckFramebufferStatusEXT", glCheckFramebufferStatusEXT == NULL);
  glewInfoFunc(fi, "glDeleteFramebuffersEXT", glDeleteFramebuffersEXT == NULL);
  glewInfoFunc(fi, "glDeleteRenderbuffersEXT", glDeleteRenderbuffersEXT == NULL);
  glewInfoFunc(fi, "glFramebufferRenderbufferEXT", glFramebufferRenderbufferEXT == NULL);
  glewInfoFunc(fi, "glFramebufferTexture1DEXT", glFramebufferTexture1DEXT == NULL);
  glewInfoFunc(fi, "glFramebufferTexture2DEXT", glFramebufferTexture2DEXT == NULL);
  glewInfoFunc(fi, "glFramebufferTexture3DEXT", glFramebufferTexture3DEXT == NULL);
  glewInfoFunc(fi, "glGenFramebuffersEXT", glGenFramebuffersEXT == NULL);
  glewInfoFunc(fi, "glGenRenderbuffersEXT", glGenRenderbuffersEXT == NULL);
  glewInfoFunc(fi, "glGenerateMipmapEXT", glGenerateMipmapEXT == NULL);
  glewInfoFunc(fi, "glGetFramebufferAttachmentParameterivEXT", glGetFramebufferAttachmentParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetRenderbufferParameterivEXT", glGetRenderbufferParameterivEXT == NULL);
  glewInfoFunc(fi, "glIsFramebufferEXT", glIsFramebufferEXT == NULL);
  glewInfoFunc(fi, "glIsRenderbufferEXT", glIsRenderbufferEXT == NULL);
  glewInfoFunc(fi, "glRenderbufferStorageEXT", glRenderbufferStorageEXT == NULL);
}

#endif /* GL_EXT_framebuffer_object */

#ifdef GL_EXT_framebuffer_sRGB

static void _glewInfo_GL_EXT_framebuffer_sRGB (void)
{
  glewPrintExt("GL_EXT_framebuffer_sRGB", GLEW_EXT_framebuffer_sRGB, glewIsSupported("GL_EXT_framebuffer_sRGB"), glewGetExtension("GL_EXT_framebuffer_sRGB"));
}

#endif /* GL_EXT_framebuffer_sRGB */

#ifdef GL_EXT_geometry_point_size

static void _glewInfo_GL_EXT_geometry_point_size (void)
{
  glewPrintExt("GL_EXT_geometry_point_size", GLEW_EXT_geometry_point_size, glewIsSupported("GL_EXT_geometry_point_size"), glewGetExtension("GL_EXT_geometry_point_size"));
}

#endif /* GL_EXT_geometry_point_size */

#ifdef GL_EXT_geometry_shader

static void _glewInfo_GL_EXT_geometry_shader (void)
{
  glewPrintExt("GL_EXT_geometry_shader", GLEW_EXT_geometry_shader, glewIsSupported("GL_EXT_geometry_shader"), glewGetExtension("GL_EXT_geometry_shader"));
}

#endif /* GL_EXT_geometry_shader */

#ifdef GL_EXT_geometry_shader4

static void _glewInfo_GL_EXT_geometry_shader4 (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_geometry_shader4", GLEW_EXT_geometry_shader4, glewIsSupported("GL_EXT_geometry_shader4"), glewGetExtension("GL_EXT_geometry_shader4"));

  glewInfoFunc(fi, "glFramebufferTextureEXT", glFramebufferTextureEXT == NULL);
  glewInfoFunc(fi, "glFramebufferTextureFaceEXT", glFramebufferTextureFaceEXT == NULL);
  glewInfoFunc(fi, "glProgramParameteriEXT", glProgramParameteriEXT == NULL);
}

#endif /* GL_EXT_geometry_shader4 */

#ifdef GL_EXT_gpu_program_parameters

static void _glewInfo_GL_EXT_gpu_program_parameters (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_gpu_program_parameters", GLEW_EXT_gpu_program_parameters, glewIsSupported("GL_EXT_gpu_program_parameters"), glewGetExtension("GL_EXT_gpu_program_parameters"));

  glewInfoFunc(fi, "glProgramEnvParameters4fvEXT", glProgramEnvParameters4fvEXT == NULL);
  glewInfoFunc(fi, "glProgramLocalParameters4fvEXT", glProgramLocalParameters4fvEXT == NULL);
}

#endif /* GL_EXT_gpu_program_parameters */

#ifdef GL_EXT_gpu_shader4

static void _glewInfo_GL_EXT_gpu_shader4 (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_gpu_shader4", GLEW_EXT_gpu_shader4, glewIsSupported("GL_EXT_gpu_shader4"), glewGetExtension("GL_EXT_gpu_shader4"));

  glewInfoFunc(fi, "glBindFragDataLocationEXT", glBindFragDataLocationEXT == NULL);
  glewInfoFunc(fi, "glGetFragDataLocationEXT", glGetFragDataLocationEXT == NULL);
  glewInfoFunc(fi, "glGetUniformuivEXT", glGetUniformuivEXT == NULL);
  glewInfoFunc(fi, "glGetVertexAttribIivEXT", glGetVertexAttribIivEXT == NULL);
  glewInfoFunc(fi, "glGetVertexAttribIuivEXT", glGetVertexAttribIuivEXT == NULL);
  glewInfoFunc(fi, "glUniform1uiEXT", glUniform1uiEXT == NULL);
  glewInfoFunc(fi, "glUniform1uivEXT", glUniform1uivEXT == NULL);
  glewInfoFunc(fi, "glUniform2uiEXT", glUniform2uiEXT == NULL);
  glewInfoFunc(fi, "glUniform2uivEXT", glUniform2uivEXT == NULL);
  glewInfoFunc(fi, "glUniform3uiEXT", glUniform3uiEXT == NULL);
  glewInfoFunc(fi, "glUniform3uivEXT", glUniform3uivEXT == NULL);
  glewInfoFunc(fi, "glUniform4uiEXT", glUniform4uiEXT == NULL);
  glewInfoFunc(fi, "glUniform4uivEXT", glUniform4uivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI1iEXT", glVertexAttribI1iEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI1ivEXT", glVertexAttribI1ivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI1uiEXT", glVertexAttribI1uiEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI1uivEXT", glVertexAttribI1uivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI2iEXT", glVertexAttribI2iEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI2ivEXT", glVertexAttribI2ivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI2uiEXT", glVertexAttribI2uiEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI2uivEXT", glVertexAttribI2uivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI3iEXT", glVertexAttribI3iEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI3ivEXT", glVertexAttribI3ivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI3uiEXT", glVertexAttribI3uiEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI3uivEXT", glVertexAttribI3uivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4bvEXT", glVertexAttribI4bvEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4iEXT", glVertexAttribI4iEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4ivEXT", glVertexAttribI4ivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4svEXT", glVertexAttribI4svEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4ubvEXT", glVertexAttribI4ubvEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4uiEXT", glVertexAttribI4uiEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4uivEXT", glVertexAttribI4uivEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribI4usvEXT", glVertexAttribI4usvEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribIPointerEXT", glVertexAttribIPointerEXT == NULL);
}

#endif /* GL_EXT_gpu_shader4 */

#ifdef GL_EXT_gpu_shader5

static void _glewInfo_GL_EXT_gpu_shader5 (void)
{
  glewPrintExt("GL_EXT_gpu_shader5", GLEW_EXT_gpu_shader5, glewIsSupported("GL_EXT_gpu_shader5"), glewGetExtension("GL_EXT_gpu_shader5"));
}

#endif /* GL_EXT_gpu_shader5 */

#ifdef GL_EXT_histogram

static void _glewInfo_GL_EXT_histogram (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_histogram", GLEW_EXT_histogram, glewIsSupported("GL_EXT_histogram"), glewGetExtension("GL_EXT_histogram"));

  glewInfoFunc(fi, "glGetHistogramEXT", glGetHistogramEXT == NULL);
  glewInfoFunc(fi, "glGetHistogramParameterfvEXT", glGetHistogramParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetHistogramParameterivEXT", glGetHistogramParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetMinmaxEXT", glGetMinmaxEXT == NULL);
  glewInfoFunc(fi, "glGetMinmaxParameterfvEXT", glGetMinmaxParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetMinmaxParameterivEXT", glGetMinmaxParameterivEXT == NULL);
  glewInfoFunc(fi, "glHistogramEXT", glHistogramEXT == NULL);
  glewInfoFunc(fi, "glMinmaxEXT", glMinmaxEXT == NULL);
  glewInfoFunc(fi, "glResetHistogramEXT", glResetHistogramEXT == NULL);
  glewInfoFunc(fi, "glResetMinmaxEXT", glResetMinmaxEXT == NULL);
}

#endif /* GL_EXT_histogram */

#ifdef GL_EXT_index_array_formats

static void _glewInfo_GL_EXT_index_array_formats (void)
{
  glewPrintExt("GL_EXT_index_array_formats", GLEW_EXT_index_array_formats, glewIsSupported("GL_EXT_index_array_formats"), glewGetExtension("GL_EXT_index_array_formats"));
}

#endif /* GL_EXT_index_array_formats */

#ifdef GL_EXT_index_func

static void _glewInfo_GL_EXT_index_func (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_index_func", GLEW_EXT_index_func, glewIsSupported("GL_EXT_index_func"), glewGetExtension("GL_EXT_index_func"));

  glewInfoFunc(fi, "glIndexFuncEXT", glIndexFuncEXT == NULL);
}

#endif /* GL_EXT_index_func */

#ifdef GL_EXT_index_material

static void _glewInfo_GL_EXT_index_material (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_index_material", GLEW_EXT_index_material, glewIsSupported("GL_EXT_index_material"), glewGetExtension("GL_EXT_index_material"));

  glewInfoFunc(fi, "glIndexMaterialEXT", glIndexMaterialEXT == NULL);
}

#endif /* GL_EXT_index_material */

#ifdef GL_EXT_index_texture

static void _glewInfo_GL_EXT_index_texture (void)
{
  glewPrintExt("GL_EXT_index_texture", GLEW_EXT_index_texture, glewIsSupported("GL_EXT_index_texture"), glewGetExtension("GL_EXT_index_texture"));
}

#endif /* GL_EXT_index_texture */

#ifdef GL_EXT_instanced_arrays

static void _glewInfo_GL_EXT_instanced_arrays (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_instanced_arrays", GLEW_EXT_instanced_arrays, glewIsSupported("GL_EXT_instanced_arrays"), glewGetExtension("GL_EXT_instanced_arrays"));

  glewInfoFunc(fi, "glVertexAttribDivisorEXT", glVertexAttribDivisorEXT == NULL);
}

#endif /* GL_EXT_instanced_arrays */

#ifdef GL_EXT_light_texture

static void _glewInfo_GL_EXT_light_texture (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_light_texture", GLEW_EXT_light_texture, glewIsSupported("GL_EXT_light_texture"), glewGetExtension("GL_EXT_light_texture"));

  glewInfoFunc(fi, "glApplyTextureEXT", glApplyTextureEXT == NULL);
  glewInfoFunc(fi, "glTextureLightEXT", glTextureLightEXT == NULL);
  glewInfoFunc(fi, "glTextureMaterialEXT", glTextureMaterialEXT == NULL);
}

#endif /* GL_EXT_light_texture */

#ifdef GL_EXT_map_buffer_range

static void _glewInfo_GL_EXT_map_buffer_range (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_map_buffer_range", GLEW_EXT_map_buffer_range, glewIsSupported("GL_EXT_map_buffer_range"), glewGetExtension("GL_EXT_map_buffer_range"));

  glewInfoFunc(fi, "glFlushMappedBufferRangeEXT", glFlushMappedBufferRangeEXT == NULL);
  glewInfoFunc(fi, "glMapBufferRangeEXT", glMapBufferRangeEXT == NULL);
}

#endif /* GL_EXT_map_buffer_range */

#ifdef GL_EXT_memory_object

static void _glewInfo_GL_EXT_memory_object (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_memory_object", GLEW_EXT_memory_object, glewIsSupported("GL_EXT_memory_object"), glewGetExtension("GL_EXT_memory_object"));

  glewInfoFunc(fi, "glBufferStorageMemEXT", glBufferStorageMemEXT == NULL);
  glewInfoFunc(fi, "glCreateMemoryObjectsEXT", glCreateMemoryObjectsEXT == NULL);
  glewInfoFunc(fi, "glDeleteMemoryObjectsEXT", glDeleteMemoryObjectsEXT == NULL);
  glewInfoFunc(fi, "glGetMemoryObjectParameterivEXT", glGetMemoryObjectParameterivEXT == NULL);
  glewInfoFunc(fi, "glGetUnsignedBytei_vEXT", glGetUnsignedBytei_vEXT == NULL);
  glewInfoFunc(fi, "glGetUnsignedBytevEXT", glGetUnsignedBytevEXT == NULL);
  glewInfoFunc(fi, "glIsMemoryObjectEXT", glIsMemoryObjectEXT == NULL);
  glewInfoFunc(fi, "glMemoryObjectParameterivEXT", glMemoryObjectParameterivEXT == NULL);
  glewInfoFunc(fi, "glNamedBufferStorageMemEXT", glNamedBufferStorageMemEXT == NULL);
  glewInfoFunc(fi, "glTexStorageMem1DEXT", glTexStorageMem1DEXT == NULL);
  glewInfoFunc(fi, "glTexStorageMem2DEXT", glTexStorageMem2DEXT == NULL);
  glewInfoFunc(fi, "glTexStorageMem2DMultisampleEXT", glTexStorageMem2DMultisampleEXT == NULL);
  glewInfoFunc(fi, "glTexStorageMem3DEXT", glTexStorageMem3DEXT == NULL);
  glewInfoFunc(fi, "glTexStorageMem3DMultisampleEXT", glTexStorageMem3DMultisampleEXT == NULL);
  glewInfoFunc(fi, "glTextureStorageMem1DEXT", glTextureStorageMem1DEXT == NULL);
  glewInfoFunc(fi, "glTextureStorageMem2DEXT", glTextureStorageMem2DEXT == NULL);
  glewInfoFunc(fi, "glTextureStorageMem2DMultisampleEXT", glTextureStorageMem2DMultisampleEXT == NULL);
  glewInfoFunc(fi, "glTextureStorageMem3DEXT", glTextureStorageMem3DEXT == NULL);
  glewInfoFunc(fi, "glTextureStorageMem3DMultisampleEXT", glTextureStorageMem3DMultisampleEXT == NULL);
}

#endif /* GL_EXT_memory_object */

#ifdef GL_EXT_memory_object_fd

static void _glewInfo_GL_EXT_memory_object_fd (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_memory_object_fd", GLEW_EXT_memory_object_fd, glewIsSupported("GL_EXT_memory_object_fd"), glewGetExtension("GL_EXT_memory_object_fd"));

  glewInfoFunc(fi, "glImportMemoryFdEXT", glImportMemoryFdEXT == NULL);
}

#endif /* GL_EXT_memory_object_fd */

#ifdef GL_EXT_memory_object_win32

static void _glewInfo_GL_EXT_memory_object_win32 (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_memory_object_win32", GLEW_EXT_memory_object_win32, glewIsSupported("GL_EXT_memory_object_win32"), glewGetExtension("GL_EXT_memory_object_win32"));

  glewInfoFunc(fi, "glImportMemoryWin32HandleEXT", glImportMemoryWin32HandleEXT == NULL);
  glewInfoFunc(fi, "glImportMemoryWin32NameEXT", glImportMemoryWin32NameEXT == NULL);
}

#endif /* GL_EXT_memory_object_win32 */

#ifdef GL_EXT_misc_attribute

static void _glewInfo_GL_EXT_misc_attribute (void)
{
  glewPrintExt("GL_EXT_misc_attribute", GLEW_EXT_misc_attribute, glewIsSupported("GL_EXT_misc_attribute"), glewGetExtension("GL_EXT_misc_attribute"));
}

#endif /* GL_EXT_misc_attribute */

#ifdef GL_EXT_multi_draw_arrays

static void _glewInfo_GL_EXT_multi_draw_arrays (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_multi_draw_arrays", GLEW_EXT_multi_draw_arrays, glewIsSupported("GL_EXT_multi_draw_arrays"), glewGetExtension("GL_EXT_multi_draw_arrays"));

  glewInfoFunc(fi, "glMultiDrawArraysEXT", glMultiDrawArraysEXT == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsEXT", glMultiDrawElementsEXT == NULL);
}

#endif /* GL_EXT_multi_draw_arrays */

#ifdef GL_EXT_multi_draw_indirect

static void _glewInfo_GL_EXT_multi_draw_indirect (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_multi_draw_indirect", GLEW_EXT_multi_draw_indirect, glewIsSupported("GL_EXT_multi_draw_indirect"), glewGetExtension("GL_EXT_multi_draw_indirect"));

  glewInfoFunc(fi, "glMultiDrawArraysIndirectEXT", glMultiDrawArraysIndirectEXT == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsIndirectEXT", glMultiDrawElementsIndirectEXT == NULL);
}

#endif /* GL_EXT_multi_draw_indirect */

#ifdef GL_EXT_multiple_textures

static void _glewInfo_GL_EXT_multiple_textures (void)
{
  glewPrintExt("GL_EXT_multiple_textures", GLEW_EXT_multiple_textures, glewIsSupported("GL_EXT_multiple_textures"), glewGetExtension("GL_EXT_multiple_textures"));
}

#endif /* GL_EXT_multiple_textures */

#ifdef GL_EXT_multisample

static void _glewInfo_GL_EXT_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_multisample", GLEW_EXT_multisample, glewIsSupported("GL_EXT_multisample"), glewGetExtension("GL_EXT_multisample"));

  glewInfoFunc(fi, "glSampleMaskEXT", glSampleMaskEXT == NULL);
  glewInfoFunc(fi, "glSamplePatternEXT", glSamplePatternEXT == NULL);
}

#endif /* GL_EXT_multisample */

#ifdef GL_EXT_multisample_compatibility

static void _glewInfo_GL_EXT_multisample_compatibility (void)
{
  glewPrintExt("GL_EXT_multisample_compatibility", GLEW_EXT_multisample_compatibility, glewIsSupported("GL_EXT_multisample_compatibility"), glewGetExtension("GL_EXT_multisample_compatibility"));
}

#endif /* GL_EXT_multisample_compatibility */

#ifdef GL_EXT_multisampled_render_to_texture

static void _glewInfo_GL_EXT_multisampled_render_to_texture (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_multisampled_render_to_texture", GLEW_EXT_multisampled_render_to_texture, glewIsSupported("GL_EXT_multisampled_render_to_texture"), glewGetExtension("GL_EXT_multisampled_render_to_texture"));

  glewInfoFunc(fi, "glFramebufferTexture2DMultisampleEXT", glFramebufferTexture2DMultisampleEXT == NULL);
}

#endif /* GL_EXT_multisampled_render_to_texture */

#ifdef GL_EXT_multisampled_render_to_texture2

static void _glewInfo_GL_EXT_multisampled_render_to_texture2 (void)
{
  glewPrintExt("GL_EXT_multisampled_render_to_texture2", GLEW_EXT_multisampled_render_to_texture2, glewIsSupported("GL_EXT_multisampled_render_to_texture2"), glewGetExtension("GL_EXT_multisampled_render_to_texture2"));
}

#endif /* GL_EXT_multisampled_render_to_texture2 */

#ifdef GL_EXT_multiview_draw_buffers

static void _glewInfo_GL_EXT_multiview_draw_buffers (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_multiview_draw_buffers", GLEW_EXT_multiview_draw_buffers, glewIsSupported("GL_EXT_multiview_draw_buffers"), glewGetExtension("GL_EXT_multiview_draw_buffers"));

  glewInfoFunc(fi, "glDrawBuffersIndexedEXT", glDrawBuffersIndexedEXT == NULL);
  glewInfoFunc(fi, "glGetIntegeri_vEXT", glGetIntegeri_vEXT == NULL);
  glewInfoFunc(fi, "glReadBufferIndexedEXT", glReadBufferIndexedEXT == NULL);
}

#endif /* GL_EXT_multiview_draw_buffers */

#ifdef GL_EXT_multiview_tessellation_geometry_shader

static void _glewInfo_GL_EXT_multiview_tessellation_geometry_shader (void)
{
  glewPrintExt("GL_EXT_multiview_tessellation_geometry_shader", GLEW_EXT_multiview_tessellation_geometry_shader, glewIsSupported("GL_EXT_multiview_tessellation_geometry_shader"), glewGetExtension("GL_EXT_multiview_tessellation_geometry_shader"));
}

#endif /* GL_EXT_multiview_tessellation_geometry_shader */

#ifdef GL_EXT_multiview_texture_multisample

static void _glewInfo_GL_EXT_multiview_texture_multisample (void)
{
  glewPrintExt("GL_EXT_multiview_texture_multisample", GLEW_EXT_multiview_texture_multisample, glewIsSupported("GL_EXT_multiview_texture_multisample"), glewGetExtension("GL_EXT_multiview_texture_multisample"));
}

#endif /* GL_EXT_multiview_texture_multisample */

#ifdef GL_EXT_multiview_timer_query

static void _glewInfo_GL_EXT_multiview_timer_query (void)
{
  glewPrintExt("GL_EXT_multiview_timer_query", GLEW_EXT_multiview_timer_query, glewIsSupported("GL_EXT_multiview_timer_query"), glewGetExtension("GL_EXT_multiview_timer_query"));
}

#endif /* GL_EXT_multiview_timer_query */

#ifdef GL_EXT_occlusion_query_boolean

static void _glewInfo_GL_EXT_occlusion_query_boolean (void)
{
  glewPrintExt("GL_EXT_occlusion_query_boolean", GLEW_EXT_occlusion_query_boolean, glewIsSupported("GL_EXT_occlusion_query_boolean"), glewGetExtension("GL_EXT_occlusion_query_boolean"));
}

#endif /* GL_EXT_occlusion_query_boolean */

#ifdef GL_EXT_packed_depth_stencil

static void _glewInfo_GL_EXT_packed_depth_stencil (void)
{
  glewPrintExt("GL_EXT_packed_depth_stencil", GLEW_EXT_packed_depth_stencil, glewIsSupported("GL_EXT_packed_depth_stencil"), glewGetExtension("GL_EXT_packed_depth_stencil"));
}

#endif /* GL_EXT_packed_depth_stencil */

#ifdef GL_EXT_packed_float

static void _glewInfo_GL_EXT_packed_float (void)
{
  glewPrintExt("GL_EXT_packed_float", GLEW_EXT_packed_float, glewIsSupported("GL_EXT_packed_float"), glewGetExtension("GL_EXT_packed_float"));
}

#endif /* GL_EXT_packed_float */

#ifdef GL_EXT_packed_pixels

static void _glewInfo_GL_EXT_packed_pixels (void)
{
  glewPrintExt("GL_EXT_packed_pixels", GLEW_EXT_packed_pixels, glewIsSupported("GL_EXT_packed_pixels"), glewGetExtension("GL_EXT_packed_pixels"));
}

#endif /* GL_EXT_packed_pixels */

#ifdef GL_EXT_paletted_texture

static void _glewInfo_GL_EXT_paletted_texture (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_paletted_texture", GLEW_EXT_paletted_texture, glewIsSupported("GL_EXT_paletted_texture"), glewGetExtension("GL_EXT_paletted_texture"));

  glewInfoFunc(fi, "glColorTableEXT", glColorTableEXT == NULL);
  glewInfoFunc(fi, "glGetColorTableEXT", glGetColorTableEXT == NULL);
  glewInfoFunc(fi, "glGetColorTableParameterfvEXT", glGetColorTableParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetColorTableParameterivEXT", glGetColorTableParameterivEXT == NULL);
}

#endif /* GL_EXT_paletted_texture */

#ifdef GL_EXT_pixel_buffer_object

static void _glewInfo_GL_EXT_pixel_buffer_object (void)
{
  glewPrintExt("GL_EXT_pixel_buffer_object", GLEW_EXT_pixel_buffer_object, glewIsSupported("GL_EXT_pixel_buffer_object"), glewGetExtension("GL_EXT_pixel_buffer_object"));
}

#endif /* GL_EXT_pixel_buffer_object */

#ifdef GL_EXT_pixel_transform

static void _glewInfo_GL_EXT_pixel_transform (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_pixel_transform", GLEW_EXT_pixel_transform, glewIsSupported("GL_EXT_pixel_transform"), glewGetExtension("GL_EXT_pixel_transform"));

  glewInfoFunc(fi, "glGetPixelTransformParameterfvEXT", glGetPixelTransformParameterfvEXT == NULL);
  glewInfoFunc(fi, "glGetPixelTransformParameterivEXT", glGetPixelTransformParameterivEXT == NULL);
  glewInfoFunc(fi, "glPixelTransformParameterfEXT", glPixelTransformParameterfEXT == NULL);
  glewInfoFunc(fi, "glPixelTransformParameterfvEXT", glPixelTransformParameterfvEXT == NULL);
  glewInfoFunc(fi, "glPixelTransformParameteriEXT", glPixelTransformParameteriEXT == NULL);
  glewInfoFunc(fi, "glPixelTransformParameterivEXT", glPixelTransformParameterivEXT == NULL);
}

#endif /* GL_EXT_pixel_transform */

#ifdef GL_EXT_pixel_transform_color_table

static void _glewInfo_GL_EXT_pixel_transform_color_table (void)
{
  glewPrintExt("GL_EXT_pixel_transform_color_table", GLEW_EXT_pixel_transform_color_table, glewIsSupported("GL_EXT_pixel_transform_color_table"), glewGetExtension("GL_EXT_pixel_transform_color_table"));
}

#endif /* GL_EXT_pixel_transform_color_table */

#ifdef GL_EXT_point_parameters

static void _glewInfo_GL_EXT_point_parameters (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_point_parameters", GLEW_EXT_point_parameters, glewIsSupported("GL_EXT_point_parameters"), glewGetExtension("GL_EXT_point_parameters"));

  glewInfoFunc(fi, "glPointParameterfEXT", glPointParameterfEXT == NULL);
  glewInfoFunc(fi, "glPointParameterfvEXT", glPointParameterfvEXT == NULL);
}

#endif /* GL_EXT_point_parameters */

#ifdef GL_EXT_polygon_offset

static void _glewInfo_GL_EXT_polygon_offset (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_polygon_offset", GLEW_EXT_polygon_offset, glewIsSupported("GL_EXT_polygon_offset"), glewGetExtension("GL_EXT_polygon_offset"));

  glewInfoFunc(fi, "glPolygonOffsetEXT", glPolygonOffsetEXT == NULL);
}

#endif /* GL_EXT_polygon_offset */

#ifdef GL_EXT_polygon_offset_clamp

static void _glewInfo_GL_EXT_polygon_offset_clamp (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_polygon_offset_clamp", GLEW_EXT_polygon_offset_clamp, glewIsSupported("GL_EXT_polygon_offset_clamp"), glewGetExtension("GL_EXT_polygon_offset_clamp"));

  glewInfoFunc(fi, "glPolygonOffsetClampEXT", glPolygonOffsetClampEXT == NULL);
}

#endif /* GL_EXT_polygon_offset_clamp */

#ifdef GL_EXT_post_depth_coverage

static void _glewInfo_GL_EXT_post_depth_coverage (void)
{
  glewPrintExt("GL_EXT_post_depth_coverage", GLEW_EXT_post_depth_coverage, glewIsSupported("GL_EXT_post_depth_coverage"), glewGetExtension("GL_EXT_post_depth_coverage"));
}

#endif /* GL_EXT_post_depth_coverage */

#ifdef GL_EXT_primitive_bounding_box

static void _glewInfo_GL_EXT_primitive_bounding_box (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_primitive_bounding_box", GLEW_EXT_primitive_bounding_box, glewIsSupported("GL_EXT_primitive_bounding_box"), glewGetExtension("GL_EXT_primitive_bounding_box"));

  glewInfoFunc(fi, "glPrimitiveBoundingBoxEXT", glPrimitiveBoundingBoxEXT == NULL);
}

#endif /* GL_EXT_primitive_bounding_box */

#ifdef GL_EXT_protected_textures

static void _glewInfo_GL_EXT_protected_textures (void)
{
  glewPrintExt("GL_EXT_protected_textures", GLEW_EXT_protected_textures, glewIsSupported("GL_EXT_protected_textures"), glewGetExtension("GL_EXT_protected_textures"));
}

#endif /* GL_EXT_protected_textures */

#ifdef GL_EXT_provoking_vertex

static void _glewInfo_GL_EXT_provoking_vertex (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_provoking_vertex", GLEW_EXT_provoking_vertex, glewIsSupported("GL_EXT_provoking_vertex"), glewGetExtension("GL_EXT_provoking_vertex"));

  glewInfoFunc(fi, "glProvokingVertexEXT", glProvokingVertexEXT == NULL);
}

#endif /* GL_EXT_provoking_vertex */

#ifdef GL_EXT_pvrtc_sRGB

static void _glewInfo_GL_EXT_pvrtc_sRGB (void)
{
  glewPrintExt("GL_EXT_pvrtc_sRGB", GLEW_EXT_pvrtc_sRGB, glewIsSupported("GL_EXT_pvrtc_sRGB"), glewGetExtension("GL_EXT_pvrtc_sRGB"));
}

#endif /* GL_EXT_pvrtc_sRGB */

#ifdef GL_EXT_raster_multisample

static void _glewInfo_GL_EXT_raster_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_raster_multisample", GLEW_EXT_raster_multisample, glewIsSupported("GL_EXT_raster_multisample"), glewGetExtension("GL_EXT_raster_multisample"));

  glewInfoFunc(fi, "glCoverageModulationNV", glCoverageModulationNV == NULL);
  glewInfoFunc(fi, "glCoverageModulationTableNV", glCoverageModulationTableNV == NULL);
  glewInfoFunc(fi, "glGetCoverageModulationTableNV", glGetCoverageModulationTableNV == NULL);
  glewInfoFunc(fi, "glRasterSamplesEXT", glRasterSamplesEXT == NULL);
}

#endif /* GL_EXT_raster_multisample */

#ifdef GL_EXT_read_format_bgra

static void _glewInfo_GL_EXT_read_format_bgra (void)
{
  glewPrintExt("GL_EXT_read_format_bgra", GLEW_EXT_read_format_bgra, glewIsSupported("GL_EXT_read_format_bgra"), glewGetExtension("GL_EXT_read_format_bgra"));
}

#endif /* GL_EXT_read_format_bgra */

#ifdef GL_EXT_render_snorm

static void _glewInfo_GL_EXT_render_snorm (void)
{
  glewPrintExt("GL_EXT_render_snorm", GLEW_EXT_render_snorm, glewIsSupported("GL_EXT_render_snorm"), glewGetExtension("GL_EXT_render_snorm"));
}

#endif /* GL_EXT_render_snorm */

#ifdef GL_EXT_rescale_normal

static void _glewInfo_GL_EXT_rescale_normal (void)
{
  glewPrintExt("GL_EXT_rescale_normal", GLEW_EXT_rescale_normal, glewIsSupported("GL_EXT_rescale_normal"), glewGetExtension("GL_EXT_rescale_normal"));
}

#endif /* GL_EXT_rescale_normal */

#ifdef GL_EXT_robustness

static void _glewInfo_GL_EXT_robustness (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_robustness", GLEW_EXT_robustness, glewIsSupported("GL_EXT_robustness"), glewGetExtension("GL_EXT_robustness"));

  glewInfoFunc(fi, "glGetnUniformfvEXT", glGetnUniformfvEXT == NULL);
  glewInfoFunc(fi, "glGetnUniformivEXT", glGetnUniformivEXT == NULL);
  glewInfoFunc(fi, "glReadnPixelsEXT", glReadnPixelsEXT == NULL);
}

#endif /* GL_EXT_robustness */

#ifdef GL_EXT_sRGB

static void _glewInfo_GL_EXT_sRGB (void)
{
  glewPrintExt("GL_EXT_sRGB", GLEW_EXT_sRGB, glewIsSupported("GL_EXT_sRGB"), glewGetExtension("GL_EXT_sRGB"));
}

#endif /* GL_EXT_sRGB */

#ifdef GL_EXT_sRGB_write_control

static void _glewInfo_GL_EXT_sRGB_write_control (void)
{
  glewPrintExt("GL_EXT_sRGB_write_control", GLEW_EXT_sRGB_write_control, glewIsSupported("GL_EXT_sRGB_write_control"), glewGetExtension("GL_EXT_sRGB_write_control"));
}

#endif /* GL_EXT_sRGB_write_control */

#ifdef GL_EXT_scene_marker

static void _glewInfo_GL_EXT_scene_marker (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_scene_marker", GLEW_EXT_scene_marker, glewIsSupported("GL_EXT_scene_marker"), glewGetExtension("GL_EXT_scene_marker"));

  glewInfoFunc(fi, "glBeginSceneEXT", glBeginSceneEXT == NULL);
  glewInfoFunc(fi, "glEndSceneEXT", glEndSceneEXT == NULL);
}

#endif /* GL_EXT_scene_marker */

#ifdef GL_EXT_secondary_color

static void _glewInfo_GL_EXT_secondary_color (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_secondary_color", GLEW_EXT_secondary_color, glewIsSupported("GL_EXT_secondary_color"), glewGetExtension("GL_EXT_secondary_color"));

  glewInfoFunc(fi, "glSecondaryColor3bEXT", glSecondaryColor3bEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3bvEXT", glSecondaryColor3bvEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3dEXT", glSecondaryColor3dEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3dvEXT", glSecondaryColor3dvEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3fEXT", glSecondaryColor3fEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3fvEXT", glSecondaryColor3fvEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3iEXT", glSecondaryColor3iEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3ivEXT", glSecondaryColor3ivEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3sEXT", glSecondaryColor3sEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3svEXT", glSecondaryColor3svEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3ubEXT", glSecondaryColor3ubEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3ubvEXT", glSecondaryColor3ubvEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3uiEXT", glSecondaryColor3uiEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3uivEXT", glSecondaryColor3uivEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3usEXT", glSecondaryColor3usEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColor3usvEXT", glSecondaryColor3usvEXT == NULL);
  glewInfoFunc(fi, "glSecondaryColorPointerEXT", glSecondaryColorPointerEXT == NULL);
}

#endif /* GL_EXT_secondary_color */

#ifdef GL_EXT_semaphore

static void _glewInfo_GL_EXT_semaphore (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_semaphore", GLEW_EXT_semaphore, glewIsSupported("GL_EXT_semaphore"), glewGetExtension("GL_EXT_semaphore"));

  glewInfoFunc(fi, "glDeleteSemaphoresEXT", glDeleteSemaphoresEXT == NULL);
  glewInfoFunc(fi, "glGenSemaphoresEXT", glGenSemaphoresEXT == NULL);
  glewInfoFunc(fi, "glGetSemaphoreParameterui64vEXT", glGetSemaphoreParameterui64vEXT == NULL);
  glewInfoFunc(fi, "glIsSemaphoreEXT", glIsSemaphoreEXT == NULL);
  glewInfoFunc(fi, "glSemaphoreParameterui64vEXT", glSemaphoreParameterui64vEXT == NULL);
  glewInfoFunc(fi, "glSignalSemaphoreEXT", glSignalSemaphoreEXT == NULL);
  glewInfoFunc(fi, "glWaitSemaphoreEXT", glWaitSemaphoreEXT == NULL);
}

#endif /* GL_EXT_semaphore */

#ifdef GL_EXT_semaphore_fd

static void _glewInfo_GL_EXT_semaphore_fd (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_semaphore_fd", GLEW_EXT_semaphore_fd, glewIsSupported("GL_EXT_semaphore_fd"), glewGetExtension("GL_EXT_semaphore_fd"));

  glewInfoFunc(fi, "glImportSemaphoreFdEXT", glImportSemaphoreFdEXT == NULL);
}

#endif /* GL_EXT_semaphore_fd */

#ifdef GL_EXT_semaphore_win32

static void _glewInfo_GL_EXT_semaphore_win32 (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_semaphore_win32", GLEW_EXT_semaphore_win32, glewIsSupported("GL_EXT_semaphore_win32"), glewGetExtension("GL_EXT_semaphore_win32"));

  glewInfoFunc(fi, "glImportSemaphoreWin32HandleEXT", glImportSemaphoreWin32HandleEXT == NULL);
  glewInfoFunc(fi, "glImportSemaphoreWin32NameEXT", glImportSemaphoreWin32NameEXT == NULL);
}

#endif /* GL_EXT_semaphore_win32 */

#ifdef GL_EXT_separate_shader_objects

static void _glewInfo_GL_EXT_separate_shader_objects (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_separate_shader_objects", GLEW_EXT_separate_shader_objects, glewIsSupported("GL_EXT_separate_shader_objects"), glewGetExtension("GL_EXT_separate_shader_objects"));

  glewInfoFunc(fi, "glActiveProgramEXT", glActiveProgramEXT == NULL);
  glewInfoFunc(fi, "glCreateShaderProgramEXT", glCreateShaderProgramEXT == NULL);
  glewInfoFunc(fi, "glUseShaderProgramEXT", glUseShaderProgramEXT == NULL);
}

#endif /* GL_EXT_separate_shader_objects */

#ifdef GL_EXT_separate_specular_color

static void _glewInfo_GL_EXT_separate_specular_color (void)
{
  glewPrintExt("GL_EXT_separate_specular_color", GLEW_EXT_separate_specular_color, glewIsSupported("GL_EXT_separate_specular_color"), glewGetExtension("GL_EXT_separate_specular_color"));
}

#endif /* GL_EXT_separate_specular_color */

#ifdef GL_EXT_shader_framebuffer_fetch

static void _glewInfo_GL_EXT_shader_framebuffer_fetch (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_shader_framebuffer_fetch", GLEW_EXT_shader_framebuffer_fetch, glewIsSupported("GL_EXT_shader_framebuffer_fetch"), glewGetExtension("GL_EXT_shader_framebuffer_fetch"));

  glewInfoFunc(fi, "glFramebufferFetchBarrierEXT", glFramebufferFetchBarrierEXT == NULL);
}

#endif /* GL_EXT_shader_framebuffer_fetch */

#ifdef GL_EXT_shader_framebuffer_fetch_non_coherent

static void _glewInfo_GL_EXT_shader_framebuffer_fetch_non_coherent (void)
{
  glewPrintExt("GL_EXT_shader_framebuffer_fetch_non_coherent", GLEW_EXT_shader_framebuffer_fetch_non_coherent, glewIsSupported("GL_EXT_shader_framebuffer_fetch_non_coherent"), glewGetExtension("GL_EXT_shader_framebuffer_fetch_non_coherent"));
}

#endif /* GL_EXT_shader_framebuffer_fetch_non_coherent */

#ifdef GL_EXT_shader_group_vote

static void _glewInfo_GL_EXT_shader_group_vote (void)
{
  glewPrintExt("GL_EXT_shader_group_vote", GLEW_EXT_shader_group_vote, glewIsSupported("GL_EXT_shader_group_vote"), glewGetExtension("GL_EXT_shader_group_vote"));
}

#endif /* GL_EXT_shader_group_vote */

#ifdef GL_EXT_shader_image_load_formatted

static void _glewInfo_GL_EXT_shader_image_load_formatted (void)
{
  glewPrintExt("GL_EXT_shader_image_load_formatted", GLEW_EXT_shader_image_load_formatted, glewIsSupported("GL_EXT_shader_image_load_formatted"), glewGetExtension("GL_EXT_shader_image_load_formatted"));
}

#endif /* GL_EXT_shader_image_load_formatted */

#ifdef GL_EXT_shader_image_load_store

static void _glewInfo_GL_EXT_shader_image_load_store (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_shader_image_load_store", GLEW_EXT_shader_image_load_store, glewIsSupported("GL_EXT_shader_image_load_store"), glewGetExtension("GL_EXT_shader_image_load_store"));

  glewInfoFunc(fi, "glBindImageTextureEXT", glBindImageTextureEXT == NULL);
  glewInfoFunc(fi, "glMemoryBarrierEXT", glMemoryBarrierEXT == NULL);
}

#endif /* GL_EXT_shader_image_load_store */

#ifdef GL_EXT_shader_implicit_conversions

static void _glewInfo_GL_EXT_shader_implicit_conversions (void)
{
  glewPrintExt("GL_EXT_shader_implicit_conversions", GLEW_EXT_shader_implicit_conversions, glewIsSupported("GL_EXT_shader_implicit_conversions"), glewGetExtension("GL_EXT_shader_implicit_conversions"));
}

#endif /* GL_EXT_shader_implicit_conversions */

#ifdef GL_EXT_shader_integer_mix

static void _glewInfo_GL_EXT_shader_integer_mix (void)
{
  glewPrintExt("GL_EXT_shader_integer_mix", GLEW_EXT_shader_integer_mix, glewIsSupported("GL_EXT_shader_integer_mix"), glewGetExtension("GL_EXT_shader_integer_mix"));
}

#endif /* GL_EXT_shader_integer_mix */

#ifdef GL_EXT_shader_io_blocks

static void _glewInfo_GL_EXT_shader_io_blocks (void)
{
  glewPrintExt("GL_EXT_shader_io_blocks", GLEW_EXT_shader_io_blocks, glewIsSupported("GL_EXT_shader_io_blocks"), glewGetExtension("GL_EXT_shader_io_blocks"));
}

#endif /* GL_EXT_shader_io_blocks */

#ifdef GL_EXT_shader_non_constant_global_initializers

static void _glewInfo_GL_EXT_shader_non_constant_global_initializers (void)
{
  glewPrintExt("GL_EXT_shader_non_constant_global_initializers", GLEW_EXT_shader_non_constant_global_initializers, glewIsSupported("GL_EXT_shader_non_constant_global_initializers"), glewGetExtension("GL_EXT_shader_non_constant_global_initializers"));
}

#endif /* GL_EXT_shader_non_constant_global_initializers */

#ifdef GL_EXT_shader_pixel_local_storage

static void _glewInfo_GL_EXT_shader_pixel_local_storage (void)
{
  glewPrintExt("GL_EXT_shader_pixel_local_storage", GLEW_EXT_shader_pixel_local_storage, glewIsSupported("GL_EXT_shader_pixel_local_storage"), glewGetExtension("GL_EXT_shader_pixel_local_storage"));
}

#endif /* GL_EXT_shader_pixel_local_storage */

#ifdef GL_EXT_shader_pixel_local_storage2

static void _glewInfo_GL_EXT_shader_pixel_local_storage2 (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_shader_pixel_local_storage2", GLEW_EXT_shader_pixel_local_storage2, glewIsSupported("GL_EXT_shader_pixel_local_storage2"), glewGetExtension("GL_EXT_shader_pixel_local_storage2"));

  glewInfoFunc(fi, "glClearPixelLocalStorageuiEXT", glClearPixelLocalStorageuiEXT == NULL);
  glewInfoFunc(fi, "glFramebufferPixelLocalStorageSizeEXT", glFramebufferPixelLocalStorageSizeEXT == NULL);
  glewInfoFunc(fi, "glGetFramebufferPixelLocalStorageSizeEXT", glGetFramebufferPixelLocalStorageSizeEXT == NULL);
}

#endif /* GL_EXT_shader_pixel_local_storage2 */

#ifdef GL_EXT_shader_texture_lod

static void _glewInfo_GL_EXT_shader_texture_lod (void)
{
  glewPrintExt("GL_EXT_shader_texture_lod", GLEW_EXT_shader_texture_lod, glewIsSupported("GL_EXT_shader_texture_lod"), glewGetExtension("GL_EXT_shader_texture_lod"));
}

#endif /* GL_EXT_shader_texture_lod */

#ifdef GL_EXT_shadow_funcs

static void _glewInfo_GL_EXT_shadow_funcs (void)
{
  glewPrintExt("GL_EXT_shadow_funcs", GLEW_EXT_shadow_funcs, glewIsSupported("GL_EXT_shadow_funcs"), glewGetExtension("GL_EXT_shadow_funcs"));
}

#endif /* GL_EXT_shadow_funcs */

#ifdef GL_EXT_shadow_samplers

static void _glewInfo_GL_EXT_shadow_samplers (void)
{
  glewPrintExt("GL_EXT_shadow_samplers", GLEW_EXT_shadow_samplers, glewIsSupported("GL_EXT_shadow_samplers"), glewGetExtension("GL_EXT_shadow_samplers"));
}

#endif /* GL_EXT_shadow_samplers */

#ifdef GL_EXT_shared_texture_palette

static void _glewInfo_GL_EXT_shared_texture_palette (void)
{
  glewPrintExt("GL_EXT_shared_texture_palette", GLEW_EXT_shared_texture_palette, glewIsSupported("GL_EXT_shared_texture_palette"), glewGetExtension("GL_EXT_shared_texture_palette"));
}

#endif /* GL_EXT_shared_texture_palette */

#ifdef GL_EXT_sparse_texture

static void _glewInfo_GL_EXT_sparse_texture (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_sparse_texture", GLEW_EXT_sparse_texture, glewIsSupported("GL_EXT_sparse_texture"), glewGetExtension("GL_EXT_sparse_texture"));

  glewInfoFunc(fi, "glTexPageCommitmentEXT", glTexPageCommitmentEXT == NULL);
  glewInfoFunc(fi, "glTexturePageCommitmentEXT", glTexturePageCommitmentEXT == NULL);
}

#endif /* GL_EXT_sparse_texture */

#ifdef GL_EXT_sparse_texture2

static void _glewInfo_GL_EXT_sparse_texture2 (void)
{
  glewPrintExt("GL_EXT_sparse_texture2", GLEW_EXT_sparse_texture2, glewIsSupported("GL_EXT_sparse_texture2"), glewGetExtension("GL_EXT_sparse_texture2"));
}

#endif /* GL_EXT_sparse_texture2 */

#ifdef GL_EXT_static_vertex_array

static void _glewInfo_GL_EXT_static_vertex_array (void)
{
  glewPrintExt("GL_EXT_static_vertex_array", GLEW_EXT_static_vertex_array, glewIsSupported("GL_EXT_static_vertex_array"), glewGetExtension("GL_EXT_static_vertex_array"));
}

#endif /* GL_EXT_static_vertex_array */

#ifdef GL_EXT_stencil_clear_tag

static void _glewInfo_GL_EXT_stencil_clear_tag (void)
{
  glewPrintExt("GL_EXT_stencil_clear_tag", GLEW_EXT_stencil_clear_tag, glewIsSupported("GL_EXT_stencil_clear_tag"), glewGetExtension("GL_EXT_stencil_clear_tag"));
}

#endif /* GL_EXT_stencil_clear_tag */

#ifdef GL_EXT_stencil_two_side

static void _glewInfo_GL_EXT_stencil_two_side (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_stencil_two_side", GLEW_EXT_stencil_two_side, glewIsSupported("GL_EXT_stencil_two_side"), glewGetExtension("GL_EXT_stencil_two_side"));

  glewInfoFunc(fi, "glActiveStencilFaceEXT", glActiveStencilFaceEXT == NULL);
}

#endif /* GL_EXT_stencil_two_side */

#ifdef GL_EXT_stencil_wrap

static void _glewInfo_GL_EXT_stencil_wrap (void)
{
  glewPrintExt("GL_EXT_stencil_wrap", GLEW_EXT_stencil_wrap, glewIsSupported("GL_EXT_stencil_wrap"), glewGetExtension("GL_EXT_stencil_wrap"));
}

#endif /* GL_EXT_stencil_wrap */

#ifdef GL_EXT_subtexture

static void _glewInfo_GL_EXT_subtexture (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_subtexture", GLEW_EXT_subtexture, glewIsSupported("GL_EXT_subtexture"), glewGetExtension("GL_EXT_subtexture"));

  glewInfoFunc(fi, "glTexSubImage1DEXT", glTexSubImage1DEXT == NULL);
  glewInfoFunc(fi, "glTexSubImage2DEXT", glTexSubImage2DEXT == NULL);
  glewInfoFunc(fi, "glTexSubImage3DEXT", glTexSubImage3DEXT == NULL);
}

#endif /* GL_EXT_subtexture */

#ifdef GL_EXT_tessellation_point_size

static void _glewInfo_GL_EXT_tessellation_point_size (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_tessellation_point_size", GLEW_EXT_tessellation_point_size, glewIsSupported("GL_EXT_tessellation_point_size"), glewGetExtension("GL_EXT_tessellation_point_size"));

  glewInfoFunc(fi, "glPatchParameteriEXT", glPatchParameteriEXT == NULL);
}

#endif /* GL_EXT_tessellation_point_size */

#ifdef GL_EXT_tessellation_shader

static void _glewInfo_GL_EXT_tessellation_shader (void)
{
  glewPrintExt("GL_EXT_tessellation_shader", GLEW_EXT_tessellation_shader, glewIsSupported("GL_EXT_tessellation_shader"), glewGetExtension("GL_EXT_tessellation_shader"));
}

#endif /* GL_EXT_tessellation_shader */

#ifdef GL_EXT_texture

static void _glewInfo_GL_EXT_texture (void)
{
  glewPrintExt("GL_EXT_texture", GLEW_EXT_texture, glewIsSupported("GL_EXT_texture"), glewGetExtension("GL_EXT_texture"));
}

#endif /* GL_EXT_texture */

#ifdef GL_EXT_texture3D

static void _glewInfo_GL_EXT_texture3D (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture3D", GLEW_EXT_texture3D, glewIsSupported("GL_EXT_texture3D"), glewGetExtension("GL_EXT_texture3D"));

  glewInfoFunc(fi, "glTexImage3DEXT", glTexImage3DEXT == NULL);
}

#endif /* GL_EXT_texture3D */

#ifdef GL_EXT_texture_array

static void _glewInfo_GL_EXT_texture_array (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_array", GLEW_EXT_texture_array, glewIsSupported("GL_EXT_texture_array"), glewGetExtension("GL_EXT_texture_array"));

  glewInfoFunc(fi, "glFramebufferTextureLayerEXT", glFramebufferTextureLayerEXT == NULL);
}

#endif /* GL_EXT_texture_array */

#ifdef GL_EXT_texture_border_clamp

static void _glewInfo_GL_EXT_texture_border_clamp (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_border_clamp", GLEW_EXT_texture_border_clamp, glewIsSupported("GL_EXT_texture_border_clamp"), glewGetExtension("GL_EXT_texture_border_clamp"));

  glewInfoFunc(fi, "glGetSamplerParameterIivEXT", glGetSamplerParameterIivEXT == NULL);
  glewInfoFunc(fi, "glGetSamplerParameterIuivEXT", glGetSamplerParameterIuivEXT == NULL);
  glewInfoFunc(fi, "glSamplerParameterIivEXT", glSamplerParameterIivEXT == NULL);
  glewInfoFunc(fi, "glSamplerParameterIuivEXT", glSamplerParameterIuivEXT == NULL);
}

#endif /* GL_EXT_texture_border_clamp */

#ifdef GL_EXT_texture_buffer

static void _glewInfo_GL_EXT_texture_buffer (void)
{
  glewPrintExt("GL_EXT_texture_buffer", GLEW_EXT_texture_buffer, glewIsSupported("GL_EXT_texture_buffer"), glewGetExtension("GL_EXT_texture_buffer"));
}

#endif /* GL_EXT_texture_buffer */

#ifdef GL_EXT_texture_buffer_object

static void _glewInfo_GL_EXT_texture_buffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_buffer_object", GLEW_EXT_texture_buffer_object, glewIsSupported("GL_EXT_texture_buffer_object"), glewGetExtension("GL_EXT_texture_buffer_object"));

  glewInfoFunc(fi, "glTexBufferEXT", glTexBufferEXT == NULL);
}

#endif /* GL_EXT_texture_buffer_object */

#ifdef GL_EXT_texture_compression_astc_decode_mode

static void _glewInfo_GL_EXT_texture_compression_astc_decode_mode (void)
{
  glewPrintExt("GL_EXT_texture_compression_astc_decode_mode", GLEW_EXT_texture_compression_astc_decode_mode, glewIsSupported("GL_EXT_texture_compression_astc_decode_mode"), glewGetExtension("GL_EXT_texture_compression_astc_decode_mode"));
}

#endif /* GL_EXT_texture_compression_astc_decode_mode */

#ifdef GL_EXT_texture_compression_astc_decode_mode_rgb9e5

static void _glewInfo_GL_EXT_texture_compression_astc_decode_mode_rgb9e5 (void)
{
  glewPrintExt("GL_EXT_texture_compression_astc_decode_mode_rgb9e5", GLEW_EXT_texture_compression_astc_decode_mode_rgb9e5, glewIsSupported("GL_EXT_texture_compression_astc_decode_mode_rgb9e5"), glewGetExtension("GL_EXT_texture_compression_astc_decode_mode_rgb9e5"));
}

#endif /* GL_EXT_texture_compression_astc_decode_mode_rgb9e5 */

#ifdef GL_EXT_texture_compression_bptc

static void _glewInfo_GL_EXT_texture_compression_bptc (void)
{
  glewPrintExt("GL_EXT_texture_compression_bptc", GLEW_EXT_texture_compression_bptc, glewIsSupported("GL_EXT_texture_compression_bptc"), glewGetExtension("GL_EXT_texture_compression_bptc"));
}

#endif /* GL_EXT_texture_compression_bptc */

#ifdef GL_EXT_texture_compression_dxt1

static void _glewInfo_GL_EXT_texture_compression_dxt1 (void)
{
  glewPrintExt("GL_EXT_texture_compression_dxt1", GLEW_EXT_texture_compression_dxt1, glewIsSupported("GL_EXT_texture_compression_dxt1"), glewGetExtension("GL_EXT_texture_compression_dxt1"));
}

#endif /* GL_EXT_texture_compression_dxt1 */

#ifdef GL_EXT_texture_compression_latc

static void _glewInfo_GL_EXT_texture_compression_latc (void)
{
  glewPrintExt("GL_EXT_texture_compression_latc", GLEW_EXT_texture_compression_latc, glewIsSupported("GL_EXT_texture_compression_latc"), glewGetExtension("GL_EXT_texture_compression_latc"));
}

#endif /* GL_EXT_texture_compression_latc */

#ifdef GL_EXT_texture_compression_rgtc

static void _glewInfo_GL_EXT_texture_compression_rgtc (void)
{
  glewPrintExt("GL_EXT_texture_compression_rgtc", GLEW_EXT_texture_compression_rgtc, glewIsSupported("GL_EXT_texture_compression_rgtc"), glewGetExtension("GL_EXT_texture_compression_rgtc"));
}

#endif /* GL_EXT_texture_compression_rgtc */

#ifdef GL_EXT_texture_compression_s3tc

static void _glewInfo_GL_EXT_texture_compression_s3tc (void)
{
  glewPrintExt("GL_EXT_texture_compression_s3tc", GLEW_EXT_texture_compression_s3tc, glewIsSupported("GL_EXT_texture_compression_s3tc"), glewGetExtension("GL_EXT_texture_compression_s3tc"));
}

#endif /* GL_EXT_texture_compression_s3tc */

#ifdef GL_EXT_texture_compression_s3tc_srgb

static void _glewInfo_GL_EXT_texture_compression_s3tc_srgb (void)
{
  glewPrintExt("GL_EXT_texture_compression_s3tc_srgb", GLEW_EXT_texture_compression_s3tc_srgb, glewIsSupported("GL_EXT_texture_compression_s3tc_srgb"), glewGetExtension("GL_EXT_texture_compression_s3tc_srgb"));
}

#endif /* GL_EXT_texture_compression_s3tc_srgb */

#ifdef GL_EXT_texture_cube_map

static void _glewInfo_GL_EXT_texture_cube_map (void)
{
  glewPrintExt("GL_EXT_texture_cube_map", GLEW_EXT_texture_cube_map, glewIsSupported("GL_EXT_texture_cube_map"), glewGetExtension("GL_EXT_texture_cube_map"));
}

#endif /* GL_EXT_texture_cube_map */

#ifdef GL_EXT_texture_cube_map_array

static void _glewInfo_GL_EXT_texture_cube_map_array (void)
{
  glewPrintExt("GL_EXT_texture_cube_map_array", GLEW_EXT_texture_cube_map_array, glewIsSupported("GL_EXT_texture_cube_map_array"), glewGetExtension("GL_EXT_texture_cube_map_array"));
}

#endif /* GL_EXT_texture_cube_map_array */

#ifdef GL_EXT_texture_edge_clamp

static void _glewInfo_GL_EXT_texture_edge_clamp (void)
{
  glewPrintExt("GL_EXT_texture_edge_clamp", GLEW_EXT_texture_edge_clamp, glewIsSupported("GL_EXT_texture_edge_clamp"), glewGetExtension("GL_EXT_texture_edge_clamp"));
}

#endif /* GL_EXT_texture_edge_clamp */

#ifdef GL_EXT_texture_env

static void _glewInfo_GL_EXT_texture_env (void)
{
  glewPrintExt("GL_EXT_texture_env", GLEW_EXT_texture_env, glewIsSupported("GL_EXT_texture_env"), glewGetExtension("GL_EXT_texture_env"));
}

#endif /* GL_EXT_texture_env */

#ifdef GL_EXT_texture_env_add

static void _glewInfo_GL_EXT_texture_env_add (void)
{
  glewPrintExt("GL_EXT_texture_env_add", GLEW_EXT_texture_env_add, glewIsSupported("GL_EXT_texture_env_add"), glewGetExtension("GL_EXT_texture_env_add"));
}

#endif /* GL_EXT_texture_env_add */

#ifdef GL_EXT_texture_env_combine

static void _glewInfo_GL_EXT_texture_env_combine (void)
{
  glewPrintExt("GL_EXT_texture_env_combine", GLEW_EXT_texture_env_combine, glewIsSupported("GL_EXT_texture_env_combine"), glewGetExtension("GL_EXT_texture_env_combine"));
}

#endif /* GL_EXT_texture_env_combine */

#ifdef GL_EXT_texture_env_dot3

static void _glewInfo_GL_EXT_texture_env_dot3 (void)
{
  glewPrintExt("GL_EXT_texture_env_dot3", GLEW_EXT_texture_env_dot3, glewIsSupported("GL_EXT_texture_env_dot3"), glewGetExtension("GL_EXT_texture_env_dot3"));
}

#endif /* GL_EXT_texture_env_dot3 */

#ifdef GL_EXT_texture_filter_anisotropic

static void _glewInfo_GL_EXT_texture_filter_anisotropic (void)
{
  glewPrintExt("GL_EXT_texture_filter_anisotropic", GLEW_EXT_texture_filter_anisotropic, glewIsSupported("GL_EXT_texture_filter_anisotropic"), glewGetExtension("GL_EXT_texture_filter_anisotropic"));
}

#endif /* GL_EXT_texture_filter_anisotropic */

#ifdef GL_EXT_texture_filter_minmax

static void _glewInfo_GL_EXT_texture_filter_minmax (void)
{
  glewPrintExt("GL_EXT_texture_filter_minmax", GLEW_EXT_texture_filter_minmax, glewIsSupported("GL_EXT_texture_filter_minmax"), glewGetExtension("GL_EXT_texture_filter_minmax"));
}

#endif /* GL_EXT_texture_filter_minmax */

#ifdef GL_EXT_texture_format_BGRA8888

static void _glewInfo_GL_EXT_texture_format_BGRA8888 (void)
{
  glewPrintExt("GL_EXT_texture_format_BGRA8888", GLEW_EXT_texture_format_BGRA8888, glewIsSupported("GL_EXT_texture_format_BGRA8888"), glewGetExtension("GL_EXT_texture_format_BGRA8888"));
}

#endif /* GL_EXT_texture_format_BGRA8888 */

#ifdef GL_EXT_texture_format_sRGB_override

static void _glewInfo_GL_EXT_texture_format_sRGB_override (void)
{
  glewPrintExt("GL_EXT_texture_format_sRGB_override", GLEW_EXT_texture_format_sRGB_override, glewIsSupported("GL_EXT_texture_format_sRGB_override"), glewGetExtension("GL_EXT_texture_format_sRGB_override"));
}

#endif /* GL_EXT_texture_format_sRGB_override */

#ifdef GL_EXT_texture_integer

static void _glewInfo_GL_EXT_texture_integer (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_integer", GLEW_EXT_texture_integer, glewIsSupported("GL_EXT_texture_integer"), glewGetExtension("GL_EXT_texture_integer"));

  glewInfoFunc(fi, "glClearColorIiEXT", glClearColorIiEXT == NULL);
  glewInfoFunc(fi, "glClearColorIuiEXT", glClearColorIuiEXT == NULL);
  glewInfoFunc(fi, "glGetTexParameterIivEXT", glGetTexParameterIivEXT == NULL);
  glewInfoFunc(fi, "glGetTexParameterIuivEXT", glGetTexParameterIuivEXT == NULL);
  glewInfoFunc(fi, "glTexParameterIivEXT", glTexParameterIivEXT == NULL);
  glewInfoFunc(fi, "glTexParameterIuivEXT", glTexParameterIuivEXT == NULL);
}

#endif /* GL_EXT_texture_integer */

#ifdef GL_EXT_texture_lod_bias

static void _glewInfo_GL_EXT_texture_lod_bias (void)
{
  glewPrintExt("GL_EXT_texture_lod_bias", GLEW_EXT_texture_lod_bias, glewIsSupported("GL_EXT_texture_lod_bias"), glewGetExtension("GL_EXT_texture_lod_bias"));
}

#endif /* GL_EXT_texture_lod_bias */

#ifdef GL_EXT_texture_mirror_clamp

static void _glewInfo_GL_EXT_texture_mirror_clamp (void)
{
  glewPrintExt("GL_EXT_texture_mirror_clamp", GLEW_EXT_texture_mirror_clamp, glewIsSupported("GL_EXT_texture_mirror_clamp"), glewGetExtension("GL_EXT_texture_mirror_clamp"));
}

#endif /* GL_EXT_texture_mirror_clamp */

#ifdef GL_EXT_texture_mirror_clamp_to_edge

static void _glewInfo_GL_EXT_texture_mirror_clamp_to_edge (void)
{
  glewPrintExt("GL_EXT_texture_mirror_clamp_to_edge", GLEW_EXT_texture_mirror_clamp_to_edge, glewIsSupported("GL_EXT_texture_mirror_clamp_to_edge"), glewGetExtension("GL_EXT_texture_mirror_clamp_to_edge"));
}

#endif /* GL_EXT_texture_mirror_clamp_to_edge */

#ifdef GL_EXT_texture_norm16

static void _glewInfo_GL_EXT_texture_norm16 (void)
{
  glewPrintExt("GL_EXT_texture_norm16", GLEW_EXT_texture_norm16, glewIsSupported("GL_EXT_texture_norm16"), glewGetExtension("GL_EXT_texture_norm16"));
}

#endif /* GL_EXT_texture_norm16 */

#ifdef GL_EXT_texture_object

static void _glewInfo_GL_EXT_texture_object (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_object", GLEW_EXT_texture_object, glewIsSupported("GL_EXT_texture_object"), glewGetExtension("GL_EXT_texture_object"));

  glewInfoFunc(fi, "glAreTexturesResidentEXT", glAreTexturesResidentEXT == NULL);
  glewInfoFunc(fi, "glBindTextureEXT", glBindTextureEXT == NULL);
  glewInfoFunc(fi, "glDeleteTexturesEXT", glDeleteTexturesEXT == NULL);
  glewInfoFunc(fi, "glGenTexturesEXT", glGenTexturesEXT == NULL);
  glewInfoFunc(fi, "glIsTextureEXT", glIsTextureEXT == NULL);
  glewInfoFunc(fi, "glPrioritizeTexturesEXT", glPrioritizeTexturesEXT == NULL);
}

#endif /* GL_EXT_texture_object */

#ifdef GL_EXT_texture_perturb_normal

static void _glewInfo_GL_EXT_texture_perturb_normal (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_perturb_normal", GLEW_EXT_texture_perturb_normal, glewIsSupported("GL_EXT_texture_perturb_normal"), glewGetExtension("GL_EXT_texture_perturb_normal"));

  glewInfoFunc(fi, "glTextureNormalEXT", glTextureNormalEXT == NULL);
}

#endif /* GL_EXT_texture_perturb_normal */

#ifdef GL_EXT_texture_query_lod

static void _glewInfo_GL_EXT_texture_query_lod (void)
{
  glewPrintExt("GL_EXT_texture_query_lod", GLEW_EXT_texture_query_lod, glewIsSupported("GL_EXT_texture_query_lod"), glewGetExtension("GL_EXT_texture_query_lod"));
}

#endif /* GL_EXT_texture_query_lod */

#ifdef GL_EXT_texture_rectangle

static void _glewInfo_GL_EXT_texture_rectangle (void)
{
  glewPrintExt("GL_EXT_texture_rectangle", GLEW_EXT_texture_rectangle, glewIsSupported("GL_EXT_texture_rectangle"), glewGetExtension("GL_EXT_texture_rectangle"));
}

#endif /* GL_EXT_texture_rectangle */

#ifdef GL_EXT_texture_rg

static void _glewInfo_GL_EXT_texture_rg (void)
{
  glewPrintExt("GL_EXT_texture_rg", GLEW_EXT_texture_rg, glewIsSupported("GL_EXT_texture_rg"), glewGetExtension("GL_EXT_texture_rg"));
}

#endif /* GL_EXT_texture_rg */

#ifdef GL_EXT_texture_sRGB

static void _glewInfo_GL_EXT_texture_sRGB (void)
{
  glewPrintExt("GL_EXT_texture_sRGB", GLEW_EXT_texture_sRGB, glewIsSupported("GL_EXT_texture_sRGB"), glewGetExtension("GL_EXT_texture_sRGB"));
}

#endif /* GL_EXT_texture_sRGB */

#ifdef GL_EXT_texture_sRGB_R8

static void _glewInfo_GL_EXT_texture_sRGB_R8 (void)
{
  glewPrintExt("GL_EXT_texture_sRGB_R8", GLEW_EXT_texture_sRGB_R8, glewIsSupported("GL_EXT_texture_sRGB_R8"), glewGetExtension("GL_EXT_texture_sRGB_R8"));
}

#endif /* GL_EXT_texture_sRGB_R8 */

#ifdef GL_EXT_texture_sRGB_RG8

static void _glewInfo_GL_EXT_texture_sRGB_RG8 (void)
{
  glewPrintExt("GL_EXT_texture_sRGB_RG8", GLEW_EXT_texture_sRGB_RG8, glewIsSupported("GL_EXT_texture_sRGB_RG8"), glewGetExtension("GL_EXT_texture_sRGB_RG8"));
}

#endif /* GL_EXT_texture_sRGB_RG8 */

#ifdef GL_EXT_texture_sRGB_decode

static void _glewInfo_GL_EXT_texture_sRGB_decode (void)
{
  glewPrintExt("GL_EXT_texture_sRGB_decode", GLEW_EXT_texture_sRGB_decode, glewIsSupported("GL_EXT_texture_sRGB_decode"), glewGetExtension("GL_EXT_texture_sRGB_decode"));
}

#endif /* GL_EXT_texture_sRGB_decode */

#ifdef GL_EXT_texture_shadow_lod

static void _glewInfo_GL_EXT_texture_shadow_lod (void)
{
  glewPrintExt("GL_EXT_texture_shadow_lod", GLEW_EXT_texture_shadow_lod, glewIsSupported("GL_EXT_texture_shadow_lod"), glewGetExtension("GL_EXT_texture_shadow_lod"));
}

#endif /* GL_EXT_texture_shadow_lod */

#ifdef GL_EXT_texture_shared_exponent

static void _glewInfo_GL_EXT_texture_shared_exponent (void)
{
  glewPrintExt("GL_EXT_texture_shared_exponent", GLEW_EXT_texture_shared_exponent, glewIsSupported("GL_EXT_texture_shared_exponent"), glewGetExtension("GL_EXT_texture_shared_exponent"));
}

#endif /* GL_EXT_texture_shared_exponent */

#ifdef GL_EXT_texture_snorm

static void _glewInfo_GL_EXT_texture_snorm (void)
{
  glewPrintExt("GL_EXT_texture_snorm", GLEW_EXT_texture_snorm, glewIsSupported("GL_EXT_texture_snorm"), glewGetExtension("GL_EXT_texture_snorm"));
}

#endif /* GL_EXT_texture_snorm */

#ifdef GL_EXT_texture_storage

static void _glewInfo_GL_EXT_texture_storage (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_storage", GLEW_EXT_texture_storage, glewIsSupported("GL_EXT_texture_storage"), glewGetExtension("GL_EXT_texture_storage"));

  glewInfoFunc(fi, "glTexStorage1DEXT", glTexStorage1DEXT == NULL);
  glewInfoFunc(fi, "glTexStorage2DEXT", glTexStorage2DEXT == NULL);
  glewInfoFunc(fi, "glTexStorage3DEXT", glTexStorage3DEXT == NULL);
  glewInfoFunc(fi, "glTextureStorage1DEXT", glTextureStorage1DEXT == NULL);
  glewInfoFunc(fi, "glTextureStorage2DEXT", glTextureStorage2DEXT == NULL);
  glewInfoFunc(fi, "glTextureStorage3DEXT", glTextureStorage3DEXT == NULL);
}

#endif /* GL_EXT_texture_storage */

#ifdef GL_EXT_texture_swizzle

static void _glewInfo_GL_EXT_texture_swizzle (void)
{
  glewPrintExt("GL_EXT_texture_swizzle", GLEW_EXT_texture_swizzle, glewIsSupported("GL_EXT_texture_swizzle"), glewGetExtension("GL_EXT_texture_swizzle"));
}

#endif /* GL_EXT_texture_swizzle */

#ifdef GL_EXT_texture_type_2_10_10_10_REV

static void _glewInfo_GL_EXT_texture_type_2_10_10_10_REV (void)
{
  glewPrintExt("GL_EXT_texture_type_2_10_10_10_REV", GLEW_EXT_texture_type_2_10_10_10_REV, glewIsSupported("GL_EXT_texture_type_2_10_10_10_REV"), glewGetExtension("GL_EXT_texture_type_2_10_10_10_REV"));
}

#endif /* GL_EXT_texture_type_2_10_10_10_REV */

#ifdef GL_EXT_texture_view

static void _glewInfo_GL_EXT_texture_view (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_texture_view", GLEW_EXT_texture_view, glewIsSupported("GL_EXT_texture_view"), glewGetExtension("GL_EXT_texture_view"));

  glewInfoFunc(fi, "glTextureViewEXT", glTextureViewEXT == NULL);
}

#endif /* GL_EXT_texture_view */

#ifdef GL_EXT_timer_query

static void _glewInfo_GL_EXT_timer_query (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_timer_query", GLEW_EXT_timer_query, glewIsSupported("GL_EXT_timer_query"), glewGetExtension("GL_EXT_timer_query"));

  glewInfoFunc(fi, "glGetQueryObjecti64vEXT", glGetQueryObjecti64vEXT == NULL);
  glewInfoFunc(fi, "glGetQueryObjectui64vEXT", glGetQueryObjectui64vEXT == NULL);
}

#endif /* GL_EXT_timer_query */

#ifdef GL_EXT_transform_feedback

static void _glewInfo_GL_EXT_transform_feedback (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_transform_feedback", GLEW_EXT_transform_feedback, glewIsSupported("GL_EXT_transform_feedback"), glewGetExtension("GL_EXT_transform_feedback"));

  glewInfoFunc(fi, "glBeginTransformFeedbackEXT", glBeginTransformFeedbackEXT == NULL);
  glewInfoFunc(fi, "glBindBufferBaseEXT", glBindBufferBaseEXT == NULL);
  glewInfoFunc(fi, "glBindBufferOffsetEXT", glBindBufferOffsetEXT == NULL);
  glewInfoFunc(fi, "glBindBufferRangeEXT", glBindBufferRangeEXT == NULL);
  glewInfoFunc(fi, "glEndTransformFeedbackEXT", glEndTransformFeedbackEXT == NULL);
  glewInfoFunc(fi, "glGetTransformFeedbackVaryingEXT", glGetTransformFeedbackVaryingEXT == NULL);
  glewInfoFunc(fi, "glTransformFeedbackVaryingsEXT", glTransformFeedbackVaryingsEXT == NULL);
}

#endif /* GL_EXT_transform_feedback */

#ifdef GL_EXT_unpack_subimage

static void _glewInfo_GL_EXT_unpack_subimage (void)
{
  glewPrintExt("GL_EXT_unpack_subimage", GLEW_EXT_unpack_subimage, glewIsSupported("GL_EXT_unpack_subimage"), glewGetExtension("GL_EXT_unpack_subimage"));
}

#endif /* GL_EXT_unpack_subimage */

#ifdef GL_EXT_vertex_array

static void _glewInfo_GL_EXT_vertex_array (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_vertex_array", GLEW_EXT_vertex_array, glewIsSupported("GL_EXT_vertex_array"), glewGetExtension("GL_EXT_vertex_array"));

  glewInfoFunc(fi, "glArrayElementEXT", glArrayElementEXT == NULL);
  glewInfoFunc(fi, "glColorPointerEXT", glColorPointerEXT == NULL);
  glewInfoFunc(fi, "glDrawArraysEXT", glDrawArraysEXT == NULL);
  glewInfoFunc(fi, "glEdgeFlagPointerEXT", glEdgeFlagPointerEXT == NULL);
  glewInfoFunc(fi, "glIndexPointerEXT", glIndexPointerEXT == NULL);
  glewInfoFunc(fi, "glNormalPointerEXT", glNormalPointerEXT == NULL);
  glewInfoFunc(fi, "glTexCoordPointerEXT", glTexCoordPointerEXT == NULL);
  glewInfoFunc(fi, "glVertexPointerEXT", glVertexPointerEXT == NULL);
}

#endif /* GL_EXT_vertex_array */

#ifdef GL_EXT_vertex_array_bgra

static void _glewInfo_GL_EXT_vertex_array_bgra (void)
{
  glewPrintExt("GL_EXT_vertex_array_bgra", GLEW_EXT_vertex_array_bgra, glewIsSupported("GL_EXT_vertex_array_bgra"), glewGetExtension("GL_EXT_vertex_array_bgra"));
}

#endif /* GL_EXT_vertex_array_bgra */

#ifdef GL_EXT_vertex_array_setXXX

static void _glewInfo_GL_EXT_vertex_array_setXXX (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_vertex_array_setXXX", GLEW_EXT_vertex_array_setXXX, glewIsSupported("GL_EXT_vertex_array_setXXX"), glewGetExtension("GL_EXT_vertex_array_setXXX"));

  glewInfoFunc(fi, "glBindArraySetEXT", glBindArraySetEXT == NULL);
  glewInfoFunc(fi, "glCreateArraySetExt", glCreateArraySetExt == NULL);
  glewInfoFunc(fi, "glDeleteArraySetsEXT", glDeleteArraySetsEXT == NULL);
}

#endif /* GL_EXT_vertex_array_setXXX */

#ifdef GL_EXT_vertex_attrib_64bit

static void _glewInfo_GL_EXT_vertex_attrib_64bit (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_vertex_attrib_64bit", GLEW_EXT_vertex_attrib_64bit, glewIsSupported("GL_EXT_vertex_attrib_64bit"), glewGetExtension("GL_EXT_vertex_attrib_64bit"));

  glewInfoFunc(fi, "glGetVertexAttribLdvEXT", glGetVertexAttribLdvEXT == NULL);
  glewInfoFunc(fi, "glVertexArrayVertexAttribLOffsetEXT", glVertexArrayVertexAttribLOffsetEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL1dEXT", glVertexAttribL1dEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL1dvEXT", glVertexAttribL1dvEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL2dEXT", glVertexAttribL2dEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL2dvEXT", glVertexAttribL2dvEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL3dEXT", glVertexAttribL3dEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL3dvEXT", glVertexAttribL3dvEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL4dEXT", glVertexAttribL4dEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribL4dvEXT", glVertexAttribL4dvEXT == NULL);
  glewInfoFunc(fi, "glVertexAttribLPointerEXT", glVertexAttribLPointerEXT == NULL);
}

#endif /* GL_EXT_vertex_attrib_64bit */

#ifdef GL_EXT_vertex_shader

static void _glewInfo_GL_EXT_vertex_shader (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_vertex_shader", GLEW_EXT_vertex_shader, glewIsSupported("GL_EXT_vertex_shader"), glewGetExtension("GL_EXT_vertex_shader"));

  glewInfoFunc(fi, "glBeginVertexShaderEXT", glBeginVertexShaderEXT == NULL);
  glewInfoFunc(fi, "glBindLightParameterEXT", glBindLightParameterEXT == NULL);
  glewInfoFunc(fi, "glBindMaterialParameterEXT", glBindMaterialParameterEXT == NULL);
  glewInfoFunc(fi, "glBindParameterEXT", glBindParameterEXT == NULL);
  glewInfoFunc(fi, "glBindTexGenParameterEXT", glBindTexGenParameterEXT == NULL);
  glewInfoFunc(fi, "glBindTextureUnitParameterEXT", glBindTextureUnitParameterEXT == NULL);
  glewInfoFunc(fi, "glBindVertexShaderEXT", glBindVertexShaderEXT == NULL);
  glewInfoFunc(fi, "glDeleteVertexShaderEXT", glDeleteVertexShaderEXT == NULL);
  glewInfoFunc(fi, "glDisableVariantClientStateEXT", glDisableVariantClientStateEXT == NULL);
  glewInfoFunc(fi, "glEnableVariantClientStateEXT", glEnableVariantClientStateEXT == NULL);
  glewInfoFunc(fi, "glEndVertexShaderEXT", glEndVertexShaderEXT == NULL);
  glewInfoFunc(fi, "glExtractComponentEXT", glExtractComponentEXT == NULL);
  glewInfoFunc(fi, "glGenSymbolsEXT", glGenSymbolsEXT == NULL);
  glewInfoFunc(fi, "glGenVertexShadersEXT", glGenVertexShadersEXT == NULL);
  glewInfoFunc(fi, "glGetInvariantBooleanvEXT", glGetInvariantBooleanvEXT == NULL);
  glewInfoFunc(fi, "glGetInvariantFloatvEXT", glGetInvariantFloatvEXT == NULL);
  glewInfoFunc(fi, "glGetInvariantIntegervEXT", glGetInvariantIntegervEXT == NULL);
  glewInfoFunc(fi, "glGetLocalConstantBooleanvEXT", glGetLocalConstantBooleanvEXT == NULL);
  glewInfoFunc(fi, "glGetLocalConstantFloatvEXT", glGetLocalConstantFloatvEXT == NULL);
  glewInfoFunc(fi, "glGetLocalConstantIntegervEXT", glGetLocalConstantIntegervEXT == NULL);
  glewInfoFunc(fi, "glGetVariantBooleanvEXT", glGetVariantBooleanvEXT == NULL);
  glewInfoFunc(fi, "glGetVariantFloatvEXT", glGetVariantFloatvEXT == NULL);
  glewInfoFunc(fi, "glGetVariantIntegervEXT", glGetVariantIntegervEXT == NULL);
  glewInfoFunc(fi, "glGetVariantPointervEXT", glGetVariantPointervEXT == NULL);
  glewInfoFunc(fi, "glInsertComponentEXT", glInsertComponentEXT == NULL);
  glewInfoFunc(fi, "glIsVariantEnabledEXT", glIsVariantEnabledEXT == NULL);
  glewInfoFunc(fi, "glSetInvariantEXT", glSetInvariantEXT == NULL);
  glewInfoFunc(fi, "glSetLocalConstantEXT", glSetLocalConstantEXT == NULL);
  glewInfoFunc(fi, "glShaderOp1EXT", glShaderOp1EXT == NULL);
  glewInfoFunc(fi, "glShaderOp2EXT", glShaderOp2EXT == NULL);
  glewInfoFunc(fi, "glShaderOp3EXT", glShaderOp3EXT == NULL);
  glewInfoFunc(fi, "glSwizzleEXT", glSwizzleEXT == NULL);
  glewInfoFunc(fi, "glVariantPointerEXT", glVariantPointerEXT == NULL);
  glewInfoFunc(fi, "glVariantbvEXT", glVariantbvEXT == NULL);
  glewInfoFunc(fi, "glVariantdvEXT", glVariantdvEXT == NULL);
  glewInfoFunc(fi, "glVariantfvEXT", glVariantfvEXT == NULL);
  glewInfoFunc(fi, "glVariantivEXT", glVariantivEXT == NULL);
  glewInfoFunc(fi, "glVariantsvEXT", glVariantsvEXT == NULL);
  glewInfoFunc(fi, "glVariantubvEXT", glVariantubvEXT == NULL);
  glewInfoFunc(fi, "glVariantuivEXT", glVariantuivEXT == NULL);
  glewInfoFunc(fi, "glVariantusvEXT", glVariantusvEXT == NULL);
  glewInfoFunc(fi, "glWriteMaskEXT", glWriteMaskEXT == NULL);
}

#endif /* GL_EXT_vertex_shader */

#ifdef GL_EXT_vertex_weighting

static void _glewInfo_GL_EXT_vertex_weighting (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_vertex_weighting", GLEW_EXT_vertex_weighting, glewIsSupported("GL_EXT_vertex_weighting"), glewGetExtension("GL_EXT_vertex_weighting"));

  glewInfoFunc(fi, "glVertexWeightPointerEXT", glVertexWeightPointerEXT == NULL);
  glewInfoFunc(fi, "glVertexWeightfEXT", glVertexWeightfEXT == NULL);
  glewInfoFunc(fi, "glVertexWeightfvEXT", glVertexWeightfvEXT == NULL);
}

#endif /* GL_EXT_vertex_weighting */

#ifdef GL_EXT_win32_keyed_mutex

static void _glewInfo_GL_EXT_win32_keyed_mutex (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_win32_keyed_mutex", GLEW_EXT_win32_keyed_mutex, glewIsSupported("GL_EXT_win32_keyed_mutex"), glewGetExtension("GL_EXT_win32_keyed_mutex"));

  glewInfoFunc(fi, "glAcquireKeyedMutexWin32EXT", glAcquireKeyedMutexWin32EXT == NULL);
  glewInfoFunc(fi, "glReleaseKeyedMutexWin32EXT", glReleaseKeyedMutexWin32EXT == NULL);
}

#endif /* GL_EXT_win32_keyed_mutex */

#ifdef GL_EXT_window_rectangles

static void _glewInfo_GL_EXT_window_rectangles (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_window_rectangles", GLEW_EXT_window_rectangles, glewIsSupported("GL_EXT_window_rectangles"), glewGetExtension("GL_EXT_window_rectangles"));

  glewInfoFunc(fi, "glWindowRectanglesEXT", glWindowRectanglesEXT == NULL);
}

#endif /* GL_EXT_window_rectangles */

#ifdef GL_EXT_x11_sync_object

static void _glewInfo_GL_EXT_x11_sync_object (void)
{
  GLboolean fi = glewPrintExt("GL_EXT_x11_sync_object", GLEW_EXT_x11_sync_object, glewIsSupported("GL_EXT_x11_sync_object"), glewGetExtension("GL_EXT_x11_sync_object"));

  glewInfoFunc(fi, "glImportSyncEXT", glImportSyncEXT == NULL);
}

#endif /* GL_EXT_x11_sync_object */

#ifdef GL_FJ_shader_binary_GCCSO

static void _glewInfo_GL_FJ_shader_binary_GCCSO (void)
{
  glewPrintExt("GL_FJ_shader_binary_GCCSO", GLEW_FJ_shader_binary_GCCSO, glewIsSupported("GL_FJ_shader_binary_GCCSO"), glewGetExtension("GL_FJ_shader_binary_GCCSO"));
}

#endif /* GL_FJ_shader_binary_GCCSO */

#ifdef GL_GREMEDY_frame_terminator

static void _glewInfo_GL_GREMEDY_frame_terminator (void)
{
  GLboolean fi = glewPrintExt("GL_GREMEDY_frame_terminator", GLEW_GREMEDY_frame_terminator, glewIsSupported("GL_GREMEDY_frame_terminator"), glewGetExtension("GL_GREMEDY_frame_terminator"));

  glewInfoFunc(fi, "glFrameTerminatorGREMEDY", glFrameTerminatorGREMEDY == NULL);
}

#endif /* GL_GREMEDY_frame_terminator */

#ifdef GL_GREMEDY_string_marker

static void _glewInfo_GL_GREMEDY_string_marker (void)
{
  GLboolean fi = glewPrintExt("GL_GREMEDY_string_marker", GLEW_GREMEDY_string_marker, glewIsSupported("GL_GREMEDY_string_marker"), glewGetExtension("GL_GREMEDY_string_marker"));

  glewInfoFunc(fi, "glStringMarkerGREMEDY", glStringMarkerGREMEDY == NULL);
}

#endif /* GL_GREMEDY_string_marker */

#ifdef GL_HP_convolution_border_modes

static void _glewInfo_GL_HP_convolution_border_modes (void)
{
  glewPrintExt("GL_HP_convolution_border_modes", GLEW_HP_convolution_border_modes, glewIsSupported("GL_HP_convolution_border_modes"), glewGetExtension("GL_HP_convolution_border_modes"));
}

#endif /* GL_HP_convolution_border_modes */

#ifdef GL_HP_image_transform

static void _glewInfo_GL_HP_image_transform (void)
{
  GLboolean fi = glewPrintExt("GL_HP_image_transform", GLEW_HP_image_transform, glewIsSupported("GL_HP_image_transform"), glewGetExtension("GL_HP_image_transform"));

  glewInfoFunc(fi, "glGetImageTransformParameterfvHP", glGetImageTransformParameterfvHP == NULL);
  glewInfoFunc(fi, "glGetImageTransformParameterivHP", glGetImageTransformParameterivHP == NULL);
  glewInfoFunc(fi, "glImageTransformParameterfHP", glImageTransformParameterfHP == NULL);
  glewInfoFunc(fi, "glImageTransformParameterfvHP", glImageTransformParameterfvHP == NULL);
  glewInfoFunc(fi, "glImageTransformParameteriHP", glImageTransformParameteriHP == NULL);
  glewInfoFunc(fi, "glImageTransformParameterivHP", glImageTransformParameterivHP == NULL);
}

#endif /* GL_HP_image_transform */

#ifdef GL_HP_occlusion_test

static void _glewInfo_GL_HP_occlusion_test (void)
{
  glewPrintExt("GL_HP_occlusion_test", GLEW_HP_occlusion_test, glewIsSupported("GL_HP_occlusion_test"), glewGetExtension("GL_HP_occlusion_test"));
}

#endif /* GL_HP_occlusion_test */

#ifdef GL_HP_texture_lighting

static void _glewInfo_GL_HP_texture_lighting (void)
{
  glewPrintExt("GL_HP_texture_lighting", GLEW_HP_texture_lighting, glewIsSupported("GL_HP_texture_lighting"), glewGetExtension("GL_HP_texture_lighting"));
}

#endif /* GL_HP_texture_lighting */

#ifdef GL_IBM_cull_vertex

static void _glewInfo_GL_IBM_cull_vertex (void)
{
  glewPrintExt("GL_IBM_cull_vertex", GLEW_IBM_cull_vertex, glewIsSupported("GL_IBM_cull_vertex"), glewGetExtension("GL_IBM_cull_vertex"));
}

#endif /* GL_IBM_cull_vertex */

#ifdef GL_IBM_multimode_draw_arrays

static void _glewInfo_GL_IBM_multimode_draw_arrays (void)
{
  GLboolean fi = glewPrintExt("GL_IBM_multimode_draw_arrays", GLEW_IBM_multimode_draw_arrays, glewIsSupported("GL_IBM_multimode_draw_arrays"), glewGetExtension("GL_IBM_multimode_draw_arrays"));

  glewInfoFunc(fi, "glMultiModeDrawArraysIBM", glMultiModeDrawArraysIBM == NULL);
  glewInfoFunc(fi, "glMultiModeDrawElementsIBM", glMultiModeDrawElementsIBM == NULL);
}

#endif /* GL_IBM_multimode_draw_arrays */

#ifdef GL_IBM_rasterpos_clip

static void _glewInfo_GL_IBM_rasterpos_clip (void)
{
  glewPrintExt("GL_IBM_rasterpos_clip", GLEW_IBM_rasterpos_clip, glewIsSupported("GL_IBM_rasterpos_clip"), glewGetExtension("GL_IBM_rasterpos_clip"));
}

#endif /* GL_IBM_rasterpos_clip */

#ifdef GL_IBM_static_data

static void _glewInfo_GL_IBM_static_data (void)
{
  glewPrintExt("GL_IBM_static_data", GLEW_IBM_static_data, glewIsSupported("GL_IBM_static_data"), glewGetExtension("GL_IBM_static_data"));
}

#endif /* GL_IBM_static_data */

#ifdef GL_IBM_texture_mirrored_repeat

static void _glewInfo_GL_IBM_texture_mirrored_repeat (void)
{
  glewPrintExt("GL_IBM_texture_mirrored_repeat", GLEW_IBM_texture_mirrored_repeat, glewIsSupported("GL_IBM_texture_mirrored_repeat"), glewGetExtension("GL_IBM_texture_mirrored_repeat"));
}

#endif /* GL_IBM_texture_mirrored_repeat */

#ifdef GL_IBM_vertex_array_lists

static void _glewInfo_GL_IBM_vertex_array_lists (void)
{
  GLboolean fi = glewPrintExt("GL_IBM_vertex_array_lists", GLEW_IBM_vertex_array_lists, glewIsSupported("GL_IBM_vertex_array_lists"), glewGetExtension("GL_IBM_vertex_array_lists"));

  glewInfoFunc(fi, "glColorPointerListIBM", glColorPointerListIBM == NULL);
  glewInfoFunc(fi, "glEdgeFlagPointerListIBM", glEdgeFlagPointerListIBM == NULL);
  glewInfoFunc(fi, "glFogCoordPointerListIBM", glFogCoordPointerListIBM == NULL);
  glewInfoFunc(fi, "glIndexPointerListIBM", glIndexPointerListIBM == NULL);
  glewInfoFunc(fi, "glNormalPointerListIBM", glNormalPointerListIBM == NULL);
  glewInfoFunc(fi, "glSecondaryColorPointerListIBM", glSecondaryColorPointerListIBM == NULL);
  glewInfoFunc(fi, "glTexCoordPointerListIBM", glTexCoordPointerListIBM == NULL);
  glewInfoFunc(fi, "glVertexPointerListIBM", glVertexPointerListIBM == NULL);
}

#endif /* GL_IBM_vertex_array_lists */

#ifdef GL_IMG_bindless_texture

static void _glewInfo_GL_IMG_bindless_texture (void)
{
  GLboolean fi = glewPrintExt("GL_IMG_bindless_texture", GLEW_IMG_bindless_texture, glewIsSupported("GL_IMG_bindless_texture"), glewGetExtension("GL_IMG_bindless_texture"));

  glewInfoFunc(fi, "glGetTextureHandleIMG", glGetTextureHandleIMG == NULL);
  glewInfoFunc(fi, "glGetTextureSamplerHandleIMG", glGetTextureSamplerHandleIMG == NULL);
  glewInfoFunc(fi, "glProgramUniformHandleui64IMG", glProgramUniformHandleui64IMG == NULL);
  glewInfoFunc(fi, "glProgramUniformHandleui64vIMG", glProgramUniformHandleui64vIMG == NULL);
  glewInfoFunc(fi, "glUniformHandleui64IMG", glUniformHandleui64IMG == NULL);
  glewInfoFunc(fi, "glUniformHandleui64vIMG", glUniformHandleui64vIMG == NULL);
}

#endif /* GL_IMG_bindless_texture */

#ifdef GL_IMG_framebuffer_downsample

static void _glewInfo_GL_IMG_framebuffer_downsample (void)
{
  GLboolean fi = glewPrintExt("GL_IMG_framebuffer_downsample", GLEW_IMG_framebuffer_downsample, glewIsSupported("GL_IMG_framebuffer_downsample"), glewGetExtension("GL_IMG_framebuffer_downsample"));

  glewInfoFunc(fi, "glFramebufferTexture2DDownsampleIMG", glFramebufferTexture2DDownsampleIMG == NULL);
  glewInfoFunc(fi, "glFramebufferTextureLayerDownsampleIMG", glFramebufferTextureLayerDownsampleIMG == NULL);
}

#endif /* GL_IMG_framebuffer_downsample */

#ifdef GL_IMG_multisampled_render_to_texture

static void _glewInfo_GL_IMG_multisampled_render_to_texture (void)
{
  GLboolean fi = glewPrintExt("GL_IMG_multisampled_render_to_texture", GLEW_IMG_multisampled_render_to_texture, glewIsSupported("GL_IMG_multisampled_render_to_texture"), glewGetExtension("GL_IMG_multisampled_render_to_texture"));

  glewInfoFunc(fi, "glFramebufferTexture2DMultisampleIMG", glFramebufferTexture2DMultisampleIMG == NULL);
  glewInfoFunc(fi, "glRenderbufferStorageMultisampleIMG", glRenderbufferStorageMultisampleIMG == NULL);
}

#endif /* GL_IMG_multisampled_render_to_texture */

#ifdef GL_IMG_program_binary

static void _glewInfo_GL_IMG_program_binary (void)
{
  glewPrintExt("GL_IMG_program_binary", GLEW_IMG_program_binary, glewIsSupported("GL_IMG_program_binary"), glewGetExtension("GL_IMG_program_binary"));
}

#endif /* GL_IMG_program_binary */

#ifdef GL_IMG_read_format

static void _glewInfo_GL_IMG_read_format (void)
{
  glewPrintExt("GL_IMG_read_format", GLEW_IMG_read_format, glewIsSupported("GL_IMG_read_format"), glewGetExtension("GL_IMG_read_format"));
}

#endif /* GL_IMG_read_format */

#ifdef GL_IMG_shader_binary

static void _glewInfo_GL_IMG_shader_binary (void)
{
  glewPrintExt("GL_IMG_shader_binary", GLEW_IMG_shader_binary, glewIsSupported("GL_IMG_shader_binary"), glewGetExtension("GL_IMG_shader_binary"));
}

#endif /* GL_IMG_shader_binary */

#ifdef GL_IMG_texture_compression_pvrtc

static void _glewInfo_GL_IMG_texture_compression_pvrtc (void)
{
  glewPrintExt("GL_IMG_texture_compression_pvrtc", GLEW_IMG_texture_compression_pvrtc, glewIsSupported("GL_IMG_texture_compression_pvrtc"), glewGetExtension("GL_IMG_texture_compression_pvrtc"));
}

#endif /* GL_IMG_texture_compression_pvrtc */

#ifdef GL_IMG_texture_compression_pvrtc2

static void _glewInfo_GL_IMG_texture_compression_pvrtc2 (void)
{
  glewPrintExt("GL_IMG_texture_compression_pvrtc2", GLEW_IMG_texture_compression_pvrtc2, glewIsSupported("GL_IMG_texture_compression_pvrtc2"), glewGetExtension("GL_IMG_texture_compression_pvrtc2"));
}

#endif /* GL_IMG_texture_compression_pvrtc2 */

#ifdef GL_IMG_texture_env_enhanced_fixed_function

static void _glewInfo_GL_IMG_texture_env_enhanced_fixed_function (void)
{
  glewPrintExt("GL_IMG_texture_env_enhanced_fixed_function", GLEW_IMG_texture_env_enhanced_fixed_function, glewIsSupported("GL_IMG_texture_env_enhanced_fixed_function"), glewGetExtension("GL_IMG_texture_env_enhanced_fixed_function"));
}

#endif /* GL_IMG_texture_env_enhanced_fixed_function */

#ifdef GL_IMG_texture_filter_cubic

static void _glewInfo_GL_IMG_texture_filter_cubic (void)
{
  glewPrintExt("GL_IMG_texture_filter_cubic", GLEW_IMG_texture_filter_cubic, glewIsSupported("GL_IMG_texture_filter_cubic"), glewGetExtension("GL_IMG_texture_filter_cubic"));
}

#endif /* GL_IMG_texture_filter_cubic */

#ifdef GL_INGR_color_clamp

static void _glewInfo_GL_INGR_color_clamp (void)
{
  glewPrintExt("GL_INGR_color_clamp", GLEW_INGR_color_clamp, glewIsSupported("GL_INGR_color_clamp"), glewGetExtension("GL_INGR_color_clamp"));
}

#endif /* GL_INGR_color_clamp */

#ifdef GL_INGR_interlace_read

static void _glewInfo_GL_INGR_interlace_read (void)
{
  glewPrintExt("GL_INGR_interlace_read", GLEW_INGR_interlace_read, glewIsSupported("GL_INGR_interlace_read"), glewGetExtension("GL_INGR_interlace_read"));
}

#endif /* GL_INGR_interlace_read */

#ifdef GL_INTEL_blackhole_render

static void _glewInfo_GL_INTEL_blackhole_render (void)
{
  glewPrintExt("GL_INTEL_blackhole_render", GLEW_INTEL_blackhole_render, glewIsSupported("GL_INTEL_blackhole_render"), glewGetExtension("GL_INTEL_blackhole_render"));
}

#endif /* GL_INTEL_blackhole_render */

#ifdef GL_INTEL_conservative_rasterization

static void _glewInfo_GL_INTEL_conservative_rasterization (void)
{
  glewPrintExt("GL_INTEL_conservative_rasterization", GLEW_INTEL_conservative_rasterization, glewIsSupported("GL_INTEL_conservative_rasterization"), glewGetExtension("GL_INTEL_conservative_rasterization"));
}

#endif /* GL_INTEL_conservative_rasterization */

#ifdef GL_INTEL_fragment_shader_ordering

static void _glewInfo_GL_INTEL_fragment_shader_ordering (void)
{
  glewPrintExt("GL_INTEL_fragment_shader_ordering", GLEW_INTEL_fragment_shader_ordering, glewIsSupported("GL_INTEL_fragment_shader_ordering"), glewGetExtension("GL_INTEL_fragment_shader_ordering"));
}

#endif /* GL_INTEL_fragment_shader_ordering */

#ifdef GL_INTEL_framebuffer_CMAA

static void _glewInfo_GL_INTEL_framebuffer_CMAA (void)
{
  glewPrintExt("GL_INTEL_framebuffer_CMAA", GLEW_INTEL_framebuffer_CMAA, glewIsSupported("GL_INTEL_framebuffer_CMAA"), glewGetExtension("GL_INTEL_framebuffer_CMAA"));
}

#endif /* GL_INTEL_framebuffer_CMAA */

#ifdef GL_INTEL_map_texture

static void _glewInfo_GL_INTEL_map_texture (void)
{
  GLboolean fi = glewPrintExt("GL_INTEL_map_texture", GLEW_INTEL_map_texture, glewIsSupported("GL_INTEL_map_texture"), glewGetExtension("GL_INTEL_map_texture"));

  glewInfoFunc(fi, "glMapTexture2DINTEL", glMapTexture2DINTEL == NULL);
  glewInfoFunc(fi, "glSyncTextureINTEL", glSyncTextureINTEL == NULL);
  glewInfoFunc(fi, "glUnmapTexture2DINTEL", glUnmapTexture2DINTEL == NULL);
}

#endif /* GL_INTEL_map_texture */

#ifdef GL_INTEL_parallel_arrays

static void _glewInfo_GL_INTEL_parallel_arrays (void)
{
  GLboolean fi = glewPrintExt("GL_INTEL_parallel_arrays", GLEW_INTEL_parallel_arrays, glewIsSupported("GL_INTEL_parallel_arrays"), glewGetExtension("GL_INTEL_parallel_arrays"));

  glewInfoFunc(fi, "glColorPointervINTEL", glColorPointervINTEL == NULL);
  glewInfoFunc(fi, "glNormalPointervINTEL", glNormalPointervINTEL == NULL);
  glewInfoFunc(fi, "glTexCoordPointervINTEL", glTexCoordPointervINTEL == NULL);
  glewInfoFunc(fi, "glVertexPointervINTEL", glVertexPointervINTEL == NULL);
}

#endif /* GL_INTEL_parallel_arrays */

#ifdef GL_INTEL_performance_query

static void _glewInfo_GL_INTEL_performance_query (void)
{
  GLboolean fi = glewPrintExt("GL_INTEL_performance_query", GLEW_INTEL_performance_query, glewIsSupported("GL_INTEL_performance_query"), glewGetExtension("GL_INTEL_performance_query"));

  glewInfoFunc(fi, "glBeginPerfQueryINTEL", glBeginPerfQueryINTEL == NULL);
  glewInfoFunc(fi, "glCreatePerfQueryINTEL", glCreatePerfQueryINTEL == NULL);
  glewInfoFunc(fi, "glDeletePerfQueryINTEL", glDeletePerfQueryINTEL == NULL);
  glewInfoFunc(fi, "glEndPerfQueryINTEL", glEndPerfQueryINTEL == NULL);
  glewInfoFunc(fi, "glGetFirstPerfQueryIdINTEL", glGetFirstPerfQueryIdINTEL == NULL);
  glewInfoFunc(fi, "glGetNextPerfQueryIdINTEL", glGetNextPerfQueryIdINTEL == NULL);
  glewInfoFunc(fi, "glGetPerfCounterInfoINTEL", glGetPerfCounterInfoINTEL == NULL);
  glewInfoFunc(fi, "glGetPerfQueryDataINTEL", glGetPerfQueryDataINTEL == NULL);
  glewInfoFunc(fi, "glGetPerfQueryIdByNameINTEL", glGetPerfQueryIdByNameINTEL == NULL);
  glewInfoFunc(fi, "glGetPerfQueryInfoINTEL", glGetPerfQueryInfoINTEL == NULL);
}

#endif /* GL_INTEL_performance_query */

#ifdef GL_INTEL_shader_integer_functions2

static void _glewInfo_GL_INTEL_shader_integer_functions2 (void)
{
  glewPrintExt("GL_INTEL_shader_integer_functions2", GLEW_INTEL_shader_integer_functions2, glewIsSupported("GL_INTEL_shader_integer_functions2"), glewGetExtension("GL_INTEL_shader_integer_functions2"));
}

#endif /* GL_INTEL_shader_integer_functions2 */

#ifdef GL_INTEL_texture_scissor

static void _glewInfo_GL_INTEL_texture_scissor (void)
{
  GLboolean fi = glewPrintExt("GL_INTEL_texture_scissor", GLEW_INTEL_texture_scissor, glewIsSupported("GL_INTEL_texture_scissor"), glewGetExtension("GL_INTEL_texture_scissor"));

  glewInfoFunc(fi, "glTexScissorFuncINTEL", glTexScissorFuncINTEL == NULL);
  glewInfoFunc(fi, "glTexScissorINTEL", glTexScissorINTEL == NULL);
}

#endif /* GL_INTEL_texture_scissor */

#ifdef GL_KHR_blend_equation_advanced

static void _glewInfo_GL_KHR_blend_equation_advanced (void)
{
  GLboolean fi = glewPrintExt("GL_KHR_blend_equation_advanced", GLEW_KHR_blend_equation_advanced, glewIsSupported("GL_KHR_blend_equation_advanced"), glewGetExtension("GL_KHR_blend_equation_advanced"));

  glewInfoFunc(fi, "glBlendBarrierKHR", glBlendBarrierKHR == NULL);
}

#endif /* GL_KHR_blend_equation_advanced */

#ifdef GL_KHR_blend_equation_advanced_coherent

static void _glewInfo_GL_KHR_blend_equation_advanced_coherent (void)
{
  glewPrintExt("GL_KHR_blend_equation_advanced_coherent", GLEW_KHR_blend_equation_advanced_coherent, glewIsSupported("GL_KHR_blend_equation_advanced_coherent"), glewGetExtension("GL_KHR_blend_equation_advanced_coherent"));
}

#endif /* GL_KHR_blend_equation_advanced_coherent */

#ifdef GL_KHR_context_flush_control

static void _glewInfo_GL_KHR_context_flush_control (void)
{
  glewPrintExt("GL_KHR_context_flush_control", GLEW_KHR_context_flush_control, glewIsSupported("GL_KHR_context_flush_control"), glewGetExtension("GL_KHR_context_flush_control"));
}

#endif /* GL_KHR_context_flush_control */

#ifdef GL_KHR_debug

static void _glewInfo_GL_KHR_debug (void)
{
  GLboolean fi = glewPrintExt("GL_KHR_debug", GLEW_KHR_debug, glewIsSupported("GL_KHR_debug"), glewGetExtension("GL_KHR_debug"));

  glewInfoFunc(fi, "glDebugMessageCallback", glDebugMessageCallback == NULL);
  glewInfoFunc(fi, "glDebugMessageControl", glDebugMessageControl == NULL);
  glewInfoFunc(fi, "glDebugMessageInsert", glDebugMessageInsert == NULL);
  glewInfoFunc(fi, "glGetDebugMessageLog", glGetDebugMessageLog == NULL);
  glewInfoFunc(fi, "glGetObjectLabel", glGetObjectLabel == NULL);
  glewInfoFunc(fi, "glGetObjectPtrLabel", glGetObjectPtrLabel == NULL);
  glewInfoFunc(fi, "glObjectLabel", glObjectLabel == NULL);
  glewInfoFunc(fi, "glObjectPtrLabel", glObjectPtrLabel == NULL);
  glewInfoFunc(fi, "glPopDebugGroup", glPopDebugGroup == NULL);
  glewInfoFunc(fi, "glPushDebugGroup", glPushDebugGroup == NULL);
}

#endif /* GL_KHR_debug */

#ifdef GL_KHR_no_error

static void _glewInfo_GL_KHR_no_error (void)
{
  glewPrintExt("GL_KHR_no_error", GLEW_KHR_no_error, glewIsSupported("GL_KHR_no_error"), glewGetExtension("GL_KHR_no_error"));
}

#endif /* GL_KHR_no_error */

#ifdef GL_KHR_parallel_shader_compile

static void _glewInfo_GL_KHR_parallel_shader_compile (void)
{
  GLboolean fi = glewPrintExt("GL_KHR_parallel_shader_compile", GLEW_KHR_parallel_shader_compile, glewIsSupported("GL_KHR_parallel_shader_compile"), glewGetExtension("GL_KHR_parallel_shader_compile"));

  glewInfoFunc(fi, "glMaxShaderCompilerThreadsKHR", glMaxShaderCompilerThreadsKHR == NULL);
}

#endif /* GL_KHR_parallel_shader_compile */

#ifdef GL_KHR_robust_buffer_access_behavior

static void _glewInfo_GL_KHR_robust_buffer_access_behavior (void)
{
  glewPrintExt("GL_KHR_robust_buffer_access_behavior", GLEW_KHR_robust_buffer_access_behavior, glewIsSupported("GL_KHR_robust_buffer_access_behavior"), glewGetExtension("GL_KHR_robust_buffer_access_behavior"));
}

#endif /* GL_KHR_robust_buffer_access_behavior */

#ifdef GL_KHR_robustness

static void _glewInfo_GL_KHR_robustness (void)
{
  GLboolean fi = glewPrintExt("GL_KHR_robustness", GLEW_KHR_robustness, glewIsSupported("GL_KHR_robustness"), glewGetExtension("GL_KHR_robustness"));

  glewInfoFunc(fi, "glGetnUniformfv", glGetnUniformfv == NULL);
  glewInfoFunc(fi, "glGetnUniformiv", glGetnUniformiv == NULL);
  glewInfoFunc(fi, "glGetnUniformuiv", glGetnUniformuiv == NULL);
  glewInfoFunc(fi, "glReadnPixels", glReadnPixels == NULL);
}

#endif /* GL_KHR_robustness */

#ifdef GL_KHR_shader_subgroup

static void _glewInfo_GL_KHR_shader_subgroup (void)
{
  glewPrintExt("GL_KHR_shader_subgroup", GLEW_KHR_shader_subgroup, glewIsSupported("GL_KHR_shader_subgroup"), glewGetExtension("GL_KHR_shader_subgroup"));
}

#endif /* GL_KHR_shader_subgroup */

#ifdef GL_KHR_texture_compression_astc_hdr

static void _glewInfo_GL_KHR_texture_compression_astc_hdr (void)
{
  glewPrintExt("GL_KHR_texture_compression_astc_hdr", GLEW_KHR_texture_compression_astc_hdr, glewIsSupported("GL_KHR_texture_compression_astc_hdr"), glewGetExtension("GL_KHR_texture_compression_astc_hdr"));
}

#endif /* GL_KHR_texture_compression_astc_hdr */

#ifdef GL_KHR_texture_compression_astc_ldr

static void _glewInfo_GL_KHR_texture_compression_astc_ldr (void)
{
  glewPrintExt("GL_KHR_texture_compression_astc_ldr", GLEW_KHR_texture_compression_astc_ldr, glewIsSupported("GL_KHR_texture_compression_astc_ldr"), glewGetExtension("GL_KHR_texture_compression_astc_ldr"));
}

#endif /* GL_KHR_texture_compression_astc_ldr */

#ifdef GL_KHR_texture_compression_astc_sliced_3d

static void _glewInfo_GL_KHR_texture_compression_astc_sliced_3d (void)
{
  glewPrintExt("GL_KHR_texture_compression_astc_sliced_3d", GLEW_KHR_texture_compression_astc_sliced_3d, glewIsSupported("GL_KHR_texture_compression_astc_sliced_3d"), glewGetExtension("GL_KHR_texture_compression_astc_sliced_3d"));
}

#endif /* GL_KHR_texture_compression_astc_sliced_3d */

#ifdef GL_KTX_buffer_region

static void _glewInfo_GL_KTX_buffer_region (void)
{
  GLboolean fi = glewPrintExt("GL_KTX_buffer_region", GLEW_KTX_buffer_region, glewIsSupported("GL_KTX_buffer_region"), glewGetExtension("GL_KTX_buffer_region"));

  glewInfoFunc(fi, "glBufferRegionEnabled", glBufferRegionEnabled == NULL);
  glewInfoFunc(fi, "glDeleteBufferRegion", glDeleteBufferRegion == NULL);
  glewInfoFunc(fi, "glDrawBufferRegion", glDrawBufferRegion == NULL);
  glewInfoFunc(fi, "glNewBufferRegion", glNewBufferRegion == NULL);
  glewInfoFunc(fi, "glReadBufferRegion", glReadBufferRegion == NULL);
}

#endif /* GL_KTX_buffer_region */

#ifdef GL_MESAX_texture_stack

static void _glewInfo_GL_MESAX_texture_stack (void)
{
  glewPrintExt("GL_MESAX_texture_stack", GLEW_MESAX_texture_stack, glewIsSupported("GL_MESAX_texture_stack"), glewGetExtension("GL_MESAX_texture_stack"));
}

#endif /* GL_MESAX_texture_stack */

#ifdef GL_MESA_framebuffer_flip_y

static void _glewInfo_GL_MESA_framebuffer_flip_y (void)
{
  GLboolean fi = glewPrintExt("GL_MESA_framebuffer_flip_y", GLEW_MESA_framebuffer_flip_y, glewIsSupported("GL_MESA_framebuffer_flip_y"), glewGetExtension("GL_MESA_framebuffer_flip_y"));

  glewInfoFunc(fi, "glFramebufferParameteriMESA", glFramebufferParameteriMESA == NULL);
  glewInfoFunc(fi, "glGetFramebufferParameterivMESA", glGetFramebufferParameterivMESA == NULL);
}

#endif /* GL_MESA_framebuffer_flip_y */

#ifdef GL_MESA_pack_invert

static void _glewInfo_GL_MESA_pack_invert (void)
{
  glewPrintExt("GL_MESA_pack_invert", GLEW_MESA_pack_invert, glewIsSupported("GL_MESA_pack_invert"), glewGetExtension("GL_MESA_pack_invert"));
}

#endif /* GL_MESA_pack_invert */

#ifdef GL_MESA_program_binary_formats

static void _glewInfo_GL_MESA_program_binary_formats (void)
{
  glewPrintExt("GL_MESA_program_binary_formats", GLEW_MESA_program_binary_formats, glewIsSupported("GL_MESA_program_binary_formats"), glewGetExtension("GL_MESA_program_binary_formats"));
}

#endif /* GL_MESA_program_binary_formats */

#ifdef GL_MESA_resize_buffers

static void _glewInfo_GL_MESA_resize_buffers (void)
{
  GLboolean fi = glewPrintExt("GL_MESA_resize_buffers", GLEW_MESA_resize_buffers, glewIsSupported("GL_MESA_resize_buffers"), glewGetExtension("GL_MESA_resize_buffers"));

  glewInfoFunc(fi, "glResizeBuffersMESA", glResizeBuffersMESA == NULL);
}

#endif /* GL_MESA_resize_buffers */

#ifdef GL_MESA_shader_integer_functions

static void _glewInfo_GL_MESA_shader_integer_functions (void)
{
  glewPrintExt("GL_MESA_shader_integer_functions", GLEW_MESA_shader_integer_functions, glewIsSupported("GL_MESA_shader_integer_functions"), glewGetExtension("GL_MESA_shader_integer_functions"));
}

#endif /* GL_MESA_shader_integer_functions */

#ifdef GL_MESA_tile_raster_order

static void _glewInfo_GL_MESA_tile_raster_order (void)
{
  glewPrintExt("GL_MESA_tile_raster_order", GLEW_MESA_tile_raster_order, glewIsSupported("GL_MESA_tile_raster_order"), glewGetExtension("GL_MESA_tile_raster_order"));
}

#endif /* GL_MESA_tile_raster_order */

#ifdef GL_MESA_window_pos

static void _glewInfo_GL_MESA_window_pos (void)
{
  GLboolean fi = glewPrintExt("GL_MESA_window_pos", GLEW_MESA_window_pos, glewIsSupported("GL_MESA_window_pos"), glewGetExtension("GL_MESA_window_pos"));

  glewInfoFunc(fi, "glWindowPos2dMESA", glWindowPos2dMESA == NULL);
  glewInfoFunc(fi, "glWindowPos2dvMESA", glWindowPos2dvMESA == NULL);
  glewInfoFunc(fi, "glWindowPos2fMESA", glWindowPos2fMESA == NULL);
  glewInfoFunc(fi, "glWindowPos2fvMESA", glWindowPos2fvMESA == NULL);
  glewInfoFunc(fi, "glWindowPos2iMESA", glWindowPos2iMESA == NULL);
  glewInfoFunc(fi, "glWindowPos2ivMESA", glWindowPos2ivMESA == NULL);
  glewInfoFunc(fi, "glWindowPos2sMESA", glWindowPos2sMESA == NULL);
  glewInfoFunc(fi, "glWindowPos2svMESA", glWindowPos2svMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3dMESA", glWindowPos3dMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3dvMESA", glWindowPos3dvMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3fMESA", glWindowPos3fMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3fvMESA", glWindowPos3fvMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3iMESA", glWindowPos3iMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3ivMESA", glWindowPos3ivMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3sMESA", glWindowPos3sMESA == NULL);
  glewInfoFunc(fi, "glWindowPos3svMESA", glWindowPos3svMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4dMESA", glWindowPos4dMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4dvMESA", glWindowPos4dvMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4fMESA", glWindowPos4fMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4fvMESA", glWindowPos4fvMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4iMESA", glWindowPos4iMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4ivMESA", glWindowPos4ivMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4sMESA", glWindowPos4sMESA == NULL);
  glewInfoFunc(fi, "glWindowPos4svMESA", glWindowPos4svMESA == NULL);
}

#endif /* GL_MESA_window_pos */

#ifdef GL_MESA_ycbcr_texture

static void _glewInfo_GL_MESA_ycbcr_texture (void)
{
  glewPrintExt("GL_MESA_ycbcr_texture", GLEW_MESA_ycbcr_texture, glewIsSupported("GL_MESA_ycbcr_texture"), glewGetExtension("GL_MESA_ycbcr_texture"));
}

#endif /* GL_MESA_ycbcr_texture */

#ifdef GL_NVX_blend_equation_advanced_multi_draw_buffers

static void _glewInfo_GL_NVX_blend_equation_advanced_multi_draw_buffers (void)
{
  glewPrintExt("GL_NVX_blend_equation_advanced_multi_draw_buffers", GLEW_NVX_blend_equation_advanced_multi_draw_buffers, glewIsSupported("GL_NVX_blend_equation_advanced_multi_draw_buffers"), glewGetExtension("GL_NVX_blend_equation_advanced_multi_draw_buffers"));
}

#endif /* GL_NVX_blend_equation_advanced_multi_draw_buffers */

#ifdef GL_NVX_conditional_render

static void _glewInfo_GL_NVX_conditional_render (void)
{
  GLboolean fi = glewPrintExt("GL_NVX_conditional_render", GLEW_NVX_conditional_render, glewIsSupported("GL_NVX_conditional_render"), glewGetExtension("GL_NVX_conditional_render"));

  glewInfoFunc(fi, "glBeginConditionalRenderNVX", glBeginConditionalRenderNVX == NULL);
  glewInfoFunc(fi, "glEndConditionalRenderNVX", glEndConditionalRenderNVX == NULL);
}

#endif /* GL_NVX_conditional_render */

#ifdef GL_NVX_gpu_memory_info

static void _glewInfo_GL_NVX_gpu_memory_info (void)
{
  glewPrintExt("GL_NVX_gpu_memory_info", GLEW_NVX_gpu_memory_info, glewIsSupported("GL_NVX_gpu_memory_info"), glewGetExtension("GL_NVX_gpu_memory_info"));
}

#endif /* GL_NVX_gpu_memory_info */

#ifdef GL_NVX_gpu_multicast2

static void _glewInfo_GL_NVX_gpu_multicast2 (void)
{
  GLboolean fi = glewPrintExt("GL_NVX_gpu_multicast2", GLEW_NVX_gpu_multicast2, glewIsSupported("GL_NVX_gpu_multicast2"), glewGetExtension("GL_NVX_gpu_multicast2"));

  glewInfoFunc(fi, "glAsyncCopyBufferSubDataNVX", glAsyncCopyBufferSubDataNVX == NULL);
  glewInfoFunc(fi, "glAsyncCopyImageSubDataNVX", glAsyncCopyImageSubDataNVX == NULL);
  glewInfoFunc(fi, "glMulticastScissorArrayvNVX", glMulticastScissorArrayvNVX == NULL);
  glewInfoFunc(fi, "glMulticastViewportArrayvNVX", glMulticastViewportArrayvNVX == NULL);
  glewInfoFunc(fi, "glMulticastViewportPositionWScaleNVX", glMulticastViewportPositionWScaleNVX == NULL);
  glewInfoFunc(fi, "glUploadGpuMaskNVX", glUploadGpuMaskNVX == NULL);
}

#endif /* GL_NVX_gpu_multicast2 */

#ifdef GL_NVX_linked_gpu_multicast

static void _glewInfo_GL_NVX_linked_gpu_multicast (void)
{
  GLboolean fi = glewPrintExt("GL_NVX_linked_gpu_multicast", GLEW_NVX_linked_gpu_multicast, glewIsSupported("GL_NVX_linked_gpu_multicast"), glewGetExtension("GL_NVX_linked_gpu_multicast"));

  glewInfoFunc(fi, "glLGPUCopyImageSubDataNVX", glLGPUCopyImageSubDataNVX == NULL);
  glewInfoFunc(fi, "glLGPUInterlockNVX", glLGPUInterlockNVX == NULL);
  glewInfoFunc(fi, "glLGPUNamedBufferSubDataNVX", glLGPUNamedBufferSubDataNVX == NULL);
}

#endif /* GL_NVX_linked_gpu_multicast */

#ifdef GL_NVX_progress_fence

static void _glewInfo_GL_NVX_progress_fence (void)
{
  GLboolean fi = glewPrintExt("GL_NVX_progress_fence", GLEW_NVX_progress_fence, glewIsSupported("GL_NVX_progress_fence"), glewGetExtension("GL_NVX_progress_fence"));

  glewInfoFunc(fi, "glClientWaitSemaphoreui64NVX", glClientWaitSemaphoreui64NVX == NULL);
  glewInfoFunc(fi, "glSignalSemaphoreui64NVX", glSignalSemaphoreui64NVX == NULL);
  glewInfoFunc(fi, "glWaitSemaphoreui64NVX", glWaitSemaphoreui64NVX == NULL);
}

#endif /* GL_NVX_progress_fence */

#ifdef GL_NV_3dvision_settings

static void _glewInfo_GL_NV_3dvision_settings (void)
{
  GLboolean fi = glewPrintExt("GL_NV_3dvision_settings", GLEW_NV_3dvision_settings, glewIsSupported("GL_NV_3dvision_settings"), glewGetExtension("GL_NV_3dvision_settings"));

  glewInfoFunc(fi, "glStereoParameterfNV", glStereoParameterfNV == NULL);
  glewInfoFunc(fi, "glStereoParameteriNV", glStereoParameteriNV == NULL);
}

#endif /* GL_NV_3dvision_settings */

#ifdef GL_NV_EGL_stream_consumer_external

static void _glewInfo_GL_NV_EGL_stream_consumer_external (void)
{
  glewPrintExt("GL_NV_EGL_stream_consumer_external", GLEW_NV_EGL_stream_consumer_external, glewIsSupported("GL_NV_EGL_stream_consumer_external"), glewGetExtension("GL_NV_EGL_stream_consumer_external"));
}

#endif /* GL_NV_EGL_stream_consumer_external */

#ifdef GL_NV_alpha_to_coverage_dither_control

static void _glewInfo_GL_NV_alpha_to_coverage_dither_control (void)
{
  GLboolean fi = glewPrintExt("GL_NV_alpha_to_coverage_dither_control", GLEW_NV_alpha_to_coverage_dither_control, glewIsSupported("GL_NV_alpha_to_coverage_dither_control"), glewGetExtension("GL_NV_alpha_to_coverage_dither_control"));

  glewInfoFunc(fi, "glAlphaToCoverageDitherControlNV", glAlphaToCoverageDitherControlNV == NULL);
}

#endif /* GL_NV_alpha_to_coverage_dither_control */

#ifdef GL_NV_bgr

static void _glewInfo_GL_NV_bgr (void)
{
  glewPrintExt("GL_NV_bgr", GLEW_NV_bgr, glewIsSupported("GL_NV_bgr"), glewGetExtension("GL_NV_bgr"));
}

#endif /* GL_NV_bgr */

#ifdef GL_NV_bindless_multi_draw_indirect

static void _glewInfo_GL_NV_bindless_multi_draw_indirect (void)
{
  GLboolean fi = glewPrintExt("GL_NV_bindless_multi_draw_indirect", GLEW_NV_bindless_multi_draw_indirect, glewIsSupported("GL_NV_bindless_multi_draw_indirect"), glewGetExtension("GL_NV_bindless_multi_draw_indirect"));

  glewInfoFunc(fi, "glMultiDrawArraysIndirectBindlessNV", glMultiDrawArraysIndirectBindlessNV == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsIndirectBindlessNV", glMultiDrawElementsIndirectBindlessNV == NULL);
}

#endif /* GL_NV_bindless_multi_draw_indirect */

#ifdef GL_NV_bindless_multi_draw_indirect_count

static void _glewInfo_GL_NV_bindless_multi_draw_indirect_count (void)
{
  GLboolean fi = glewPrintExt("GL_NV_bindless_multi_draw_indirect_count", GLEW_NV_bindless_multi_draw_indirect_count, glewIsSupported("GL_NV_bindless_multi_draw_indirect_count"), glewGetExtension("GL_NV_bindless_multi_draw_indirect_count"));

  glewInfoFunc(fi, "glMultiDrawArraysIndirectBindlessCountNV", glMultiDrawArraysIndirectBindlessCountNV == NULL);
  glewInfoFunc(fi, "glMultiDrawElementsIndirectBindlessCountNV", glMultiDrawElementsIndirectBindlessCountNV == NULL);
}

#endif /* GL_NV_bindless_multi_draw_indirect_count */

#ifdef GL_NV_bindless_texture

static void _glewInfo_GL_NV_bindless_texture (void)
{
  GLboolean fi = glewPrintExt("GL_NV_bindless_texture", GLEW_NV_bindless_texture, glewIsSupported("GL_NV_bindless_texture"), glewGetExtension("GL_NV_bindless_texture"));

  glewInfoFunc(fi, "glGetImageHandleNV", glGetImageHandleNV == NULL);
  glewInfoFunc(fi, "glGetTextureHandleNV", glGetTextureHandleNV == NULL);
  glewInfoFunc(fi, "glGetTextureSamplerHandleNV", glGetTextureSamplerHandleNV == NULL);
  glewInfoFunc(fi, "glIsImageHandleResidentNV", glIsImageHandleResidentNV == NULL);
  glewInfoFunc(fi, "glIsTextureHandleResidentNV", glIsTextureHandleResidentNV == NULL);
  glewInfoFunc(fi, "glMakeImageHandleNonResidentNV", glMakeImageHandleNonResidentNV == NULL);
  glewInfoFunc(fi, "glMakeImageHandleResidentNV", glMakeImageHandleResidentNV == NULL);
  glewInfoFunc(fi, "glMakeTextureHandleNonResidentNV", glMakeTextureHandleNonResidentNV == NULL);
  glewInfoFunc(fi, "glMakeTextureHandleResidentNV", glMakeTextureHandleResidentNV == NULL);
  glewInfoFunc(fi, "glProgramUniformHandleui64NV", glProgramUniformHandleui64NV == NULL);
  glewInfoFunc(fi, "glProgramUniformHandleui64vNV", glProgramUniformHandleui64vNV == NULL);
  glewInfoFunc(fi, "glUniformHandleui64NV", glUniformHandleui64NV == NULL);
  glewInfoFunc(fi, "glUniformHandleui64vNV", glUniformHandleui64vNV == NULL);
}

#endif /* GL_NV_bindless_texture */

#ifdef GL_NV_blend_equation_advanced

static void _glewInfo_GL_NV_blend_equation_advanced (void)
{
  GLboolean fi = glewPrintExt("GL_NV_blend_equation_advanced", GLEW_NV_blend_equation_advanced, glewIsSupported("GL_NV_blend_equation_advanced"), glewGetExtension("GL_NV_blend_equation_advanced"));

  glewInfoFunc(fi, "glBlendBarrierNV", glBlendBarrierNV == NULL);
  glewInfoFunc(fi, "glBlendParameteriNV", glBlendParameteriNV == NULL);
}

#endif /* GL_NV_blend_equation_advanced */

#ifdef GL_NV_blend_equation_advanced_coherent

static void _glewInfo_GL_NV_blend_equation_advanced_coherent (void)
{
  glewPrintExt("GL_NV_blend_equation_advanced_coherent", GLEW_NV_blend_equation_advanced_coherent, glewIsSupported("GL_NV_blend_equation_advanced_coherent"), glewGetExtension("GL_NV_blend_equation_advanced_coherent"));
}

#endif /* GL_NV_blend_equation_advanced_coherent */

#ifdef GL_NV_blend_minmax_factor

static void _glewInfo_GL_NV_blend_minmax_factor (void)
{
  glewPrintExt("GL_NV_blend_minmax_factor", GLEW_NV_blend_minmax_factor, glewIsSupported("GL_NV_blend_minmax_factor"), glewGetExtension("GL_NV_blend_minmax_factor"));
}

#endif /* GL_NV_blend_minmax_factor */

#ifdef GL_NV_blend_square

static void _glewInfo_GL_NV_blend_square (void)
{
  glewPrintExt("GL_NV_blend_square", GLEW_NV_blend_square, glewIsSupported("GL_NV_blend_square"), glewGetExtension("GL_NV_blend_square"));
}

#endif /* GL_NV_blend_square */

#ifdef GL_NV_clip_space_w_scaling

static void _glewInfo_GL_NV_clip_space_w_scaling (void)
{
  GLboolean fi = glewPrintExt("GL_NV_clip_space_w_scaling", GLEW_NV_clip_space_w_scaling, glewIsSupported("GL_NV_clip_space_w_scaling"), glewGetExtension("GL_NV_clip_space_w_scaling"));

  glewInfoFunc(fi, "glViewportPositionWScaleNV", glViewportPositionWScaleNV == NULL);
}

#endif /* GL_NV_clip_space_w_scaling */

#ifdef GL_NV_command_list

static void _glewInfo_GL_NV_command_list (void)
{
  GLboolean fi = glewPrintExt("GL_NV_command_list", GLEW_NV_command_list, glewIsSupported("GL_NV_command_list"), glewGetExtension("GL_NV_command_list"));

  glewInfoFunc(fi, "glCallCommandListNV", glCallCommandListNV == NULL);
  glewInfoFunc(fi, "glCommandListSegmentsNV", glCommandListSegmentsNV == NULL);
  glewInfoFunc(fi, "glCompileCommandListNV", glCompileCommandListNV == NULL);
  glewInfoFunc(fi, "glCreateCommandListsNV", glCreateCommandListsNV == NULL);
  glewInfoFunc(fi, "glCreateStatesNV", glCreateStatesNV == NULL);
  glewInfoFunc(fi, "glDeleteCommandListsNV", glDeleteCommandListsNV == NULL);
  glewInfoFunc(fi, "glDeleteStatesNV", glDeleteStatesNV == NULL);
  glewInfoFunc(fi, "glDrawCommandsAddressNV", glDrawCommandsAddressNV == NULL);
  glewInfoFunc(fi, "glDrawCommandsNV", glDrawCommandsNV == NULL);
  glewInfoFunc(fi, "glDrawCommandsStatesAddressNV", glDrawCommandsStatesAddressNV == NULL);
  glewInfoFunc(fi, "glDrawCommandsStatesNV", glDrawCommandsStatesNV == NULL);
  glewInfoFunc(fi, "glGetCommandHeaderNV", glGetCommandHeaderNV == NULL);
  glewInfoFunc(fi, "glGetStageIndexNV", glGetStageIndexNV == NULL);
  glewInfoFunc(fi, "glIsCommandListNV", glIsCommandListNV == NULL);
  glewInfoFunc(fi, "glIsStateNV", glIsStateNV == NULL);
  glewInfoFunc(fi, "glListDrawCommandsStatesClientNV", glListDrawCommandsStatesClientNV == NULL);
  glewInfoFunc(fi, "glStateCaptureNV", glStateCaptureNV == NULL);
}

#endif /* GL_NV_command_list */

#ifdef GL_NV_compute_program5

static void _glewInfo_GL_NV_compute_program5 (void)
{
  glewPrintExt("GL_NV_compute_program5", GLEW_NV_compute_program5, glewIsSupported("GL_NV_compute_program5"), glewGetExtension("GL_NV_compute_program5"));
}

#endif /* GL_NV_compute_program5 */

#ifdef GL_NV_compute_shader_derivatives

static void _glewInfo_GL_NV_compute_shader_derivatives (void)
{
  glewPrintExt("GL_NV_compute_shader_derivatives", GLEW_NV_compute_shader_derivatives, glewIsSupported("GL_NV_compute_shader_derivatives"), glewGetExtension("GL_NV_compute_shader_derivatives"));
}

#endif /* GL_NV_compute_shader_derivatives */

#ifdef GL_NV_conditional_render

static void _glewInfo_GL_NV_conditional_render (void)
{
  GLboolean fi = glewPrintExt("GL_NV_conditional_render", GLEW_NV_conditional_render, glewIsSupported("GL_NV_conditional_render"), glewGetExtension("GL_NV_conditional_render"));

  glewInfoFunc(fi, "glBeginConditionalRenderNV", glBeginConditionalRenderNV == NULL);
  glewInfoFunc(fi, "glEndConditionalRenderNV", glEndConditionalRenderNV == NULL);
}

#endif /* GL_NV_conditional_render */

#ifdef GL_NV_conservative_raster

static void _glewInfo_GL_NV_conservative_raster (void)
{
  GLboolean fi = glewPrintExt("GL_NV_conservative_raster", GLEW_NV_conservative_raster, glewIsSupported("GL_NV_conservative_raster"), glewGetExtension("GL_NV_conservative_raster"));

  glewInfoFunc(fi, "glSubpixelPrecisionBiasNV", glSubpixelPrecisionBiasNV == NULL);
}

#endif /* GL_NV_conservative_raster */

#ifdef GL_NV_conservative_raster_dilate

static void _glewInfo_GL_NV_conservative_raster_dilate (void)
{
  GLboolean fi = glewPrintExt("GL_NV_conservative_raster_dilate", GLEW_NV_conservative_raster_dilate, glewIsSupported("GL_NV_conservative_raster_dilate"), glewGetExtension("GL_NV_conservative_raster_dilate"));

  glewInfoFunc(fi, "glConservativeRasterParameterfNV", glConservativeRasterParameterfNV == NULL);
}

#endif /* GL_NV_conservative_raster_dilate */

#ifdef GL_NV_conservative_raster_pre_snap

static void _glewInfo_GL_NV_conservative_raster_pre_snap (void)
{
  glewPrintExt("GL_NV_conservative_raster_pre_snap", GLEW_NV_conservative_raster_pre_snap, glewIsSupported("GL_NV_conservative_raster_pre_snap"), glewGetExtension("GL_NV_conservative_raster_pre_snap"));
}

#endif /* GL_NV_conservative_raster_pre_snap */

#ifdef GL_NV_conservative_raster_pre_snap_triangles

static void _glewInfo_GL_NV_conservative_raster_pre_snap_triangles (void)
{
  GLboolean fi = glewPrintExt("GL_NV_conservative_raster_pre_snap_triangles", GLEW_NV_conservative_raster_pre_snap_triangles, glewIsSupported("GL_NV_conservative_raster_pre_snap_triangles"), glewGetExtension("GL_NV_conservative_raster_pre_snap_triangles"));

  glewInfoFunc(fi, "glConservativeRasterParameteriNV", glConservativeRasterParameteriNV == NULL);
}

#endif /* GL_NV_conservative_raster_pre_snap_triangles */

#ifdef GL_NV_conservative_raster_underestimation

static void _glewInfo_GL_NV_conservative_raster_underestimation (void)
{
  glewPrintExt("GL_NV_conservative_raster_underestimation", GLEW_NV_conservative_raster_underestimation, glewIsSupported("GL_NV_conservative_raster_underestimation"), glewGetExtension("GL_NV_conservative_raster_underestimation"));
}

#endif /* GL_NV_conservative_raster_underestimation */

#ifdef GL_NV_copy_buffer

static void _glewInfo_GL_NV_copy_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_NV_copy_buffer", GLEW_NV_copy_buffer, glewIsSupported("GL_NV_copy_buffer"), glewGetExtension("GL_NV_copy_buffer"));

  glewInfoFunc(fi, "glCopyBufferSubDataNV", glCopyBufferSubDataNV == NULL);
}

#endif /* GL_NV_copy_buffer */

#ifdef GL_NV_copy_depth_to_color

static void _glewInfo_GL_NV_copy_depth_to_color (void)
{
  glewPrintExt("GL_NV_copy_depth_to_color", GLEW_NV_copy_depth_to_color, glewIsSupported("GL_NV_copy_depth_to_color"), glewGetExtension("GL_NV_copy_depth_to_color"));
}

#endif /* GL_NV_copy_depth_to_color */

#ifdef GL_NV_copy_image

static void _glewInfo_GL_NV_copy_image (void)
{
  GLboolean fi = glewPrintExt("GL_NV_copy_image", GLEW_NV_copy_image, glewIsSupported("GL_NV_copy_image"), glewGetExtension("GL_NV_copy_image"));

  glewInfoFunc(fi, "glCopyImageSubDataNV", glCopyImageSubDataNV == NULL);
}

#endif /* GL_NV_copy_image */

#ifdef GL_NV_deep_texture3D

static void _glewInfo_GL_NV_deep_texture3D (void)
{
  glewPrintExt("GL_NV_deep_texture3D", GLEW_NV_deep_texture3D, glewIsSupported("GL_NV_deep_texture3D"), glewGetExtension("GL_NV_deep_texture3D"));
}

#endif /* GL_NV_deep_texture3D */

#ifdef GL_NV_depth_buffer_float

static void _glewInfo_GL_NV_depth_buffer_float (void)
{
  GLboolean fi = glewPrintExt("GL_NV_depth_buffer_float", GLEW_NV_depth_buffer_float, glewIsSupported("GL_NV_depth_buffer_float"), glewGetExtension("GL_NV_depth_buffer_float"));

  glewInfoFunc(fi, "glClearDepthdNV", glClearDepthdNV == NULL);
  glewInfoFunc(fi, "glDepthBoundsdNV", glDepthBoundsdNV == NULL);
  glewInfoFunc(fi, "glDepthRangedNV", glDepthRangedNV == NULL);
}

#endif /* GL_NV_depth_buffer_float */

#ifdef GL_NV_depth_clamp

static void _glewInfo_GL_NV_depth_clamp (void)
{
  glewPrintExt("GL_NV_depth_clamp", GLEW_NV_depth_clamp, glewIsSupported("GL_NV_depth_clamp"), glewGetExtension("GL_NV_depth_clamp"));
}

#endif /* GL_NV_depth_clamp */

#ifdef GL_NV_depth_nonlinear

static void _glewInfo_GL_NV_depth_nonlinear (void)
{
  glewPrintExt("GL_NV_depth_nonlinear", GLEW_NV_depth_nonlinear, glewIsSupported("GL_NV_depth_nonlinear"), glewGetExtension("GL_NV_depth_nonlinear"));
}

#endif /* GL_NV_depth_nonlinear */

#ifdef GL_NV_depth_range_unclamped

static void _glewInfo_GL_NV_depth_range_unclamped (void)
{
  glewPrintExt("GL_NV_depth_range_unclamped", GLEW_NV_depth_range_unclamped, glewIsSupported("GL_NV_depth_range_unclamped"), glewGetExtension("GL_NV_depth_range_unclamped"));
}

#endif /* GL_NV_depth_range_unclamped */

#ifdef GL_NV_draw_buffers

static void _glewInfo_GL_NV_draw_buffers (void)
{
  GLboolean fi = glewPrintExt("GL_NV_draw_buffers", GLEW_NV_draw_buffers, glewIsSupported("GL_NV_draw_buffers"), glewGetExtension("GL_NV_draw_buffers"));

  glewInfoFunc(fi, "glDrawBuffersNV", glDrawBuffersNV == NULL);
}

#endif /* GL_NV_draw_buffers */

#ifdef GL_NV_draw_instanced

static void _glewInfo_GL_NV_draw_instanced (void)
{
  GLboolean fi = glewPrintExt("GL_NV_draw_instanced", GLEW_NV_draw_instanced, glewIsSupported("GL_NV_draw_instanced"), glewGetExtension("GL_NV_draw_instanced"));

  glewInfoFunc(fi, "glDrawArraysInstancedNV", glDrawArraysInstancedNV == NULL);
  glewInfoFunc(fi, "glDrawElementsInstancedNV", glDrawElementsInstancedNV == NULL);
}

#endif /* GL_NV_draw_instanced */

#ifdef GL_NV_draw_texture

static void _glewInfo_GL_NV_draw_texture (void)
{
  GLboolean fi = glewPrintExt("GL_NV_draw_texture", GLEW_NV_draw_texture, glewIsSupported("GL_NV_draw_texture"), glewGetExtension("GL_NV_draw_texture"));

  glewInfoFunc(fi, "glDrawTextureNV", glDrawTextureNV == NULL);
}

#endif /* GL_NV_draw_texture */

#ifdef GL_NV_draw_vulkan_image

static void _glewInfo_GL_NV_draw_vulkan_image (void)
{
  GLboolean fi = glewPrintExt("GL_NV_draw_vulkan_image", GLEW_NV_draw_vulkan_image, glewIsSupported("GL_NV_draw_vulkan_image"), glewGetExtension("GL_NV_draw_vulkan_image"));

  glewInfoFunc(fi, "glDrawVkImageNV", glDrawVkImageNV == NULL);
  glewInfoFunc(fi, "glGetVkProcAddrNV", glGetVkProcAddrNV == NULL);
  glewInfoFunc(fi, "glSignalVkFenceNV", glSignalVkFenceNV == NULL);
  glewInfoFunc(fi, "glSignalVkSemaphoreNV", glSignalVkSemaphoreNV == NULL);
  glewInfoFunc(fi, "glWaitVkSemaphoreNV", glWaitVkSemaphoreNV == NULL);
}

#endif /* GL_NV_draw_vulkan_image */

#ifdef GL_NV_evaluators

static void _glewInfo_GL_NV_evaluators (void)
{
  GLboolean fi = glewPrintExt("GL_NV_evaluators", GLEW_NV_evaluators, glewIsSupported("GL_NV_evaluators"), glewGetExtension("GL_NV_evaluators"));

  glewInfoFunc(fi, "glEvalMapsNV", glEvalMapsNV == NULL);
  glewInfoFunc(fi, "glGetMapAttribParameterfvNV", glGetMapAttribParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetMapAttribParameterivNV", glGetMapAttribParameterivNV == NULL);
  glewInfoFunc(fi, "glGetMapControlPointsNV", glGetMapControlPointsNV == NULL);
  glewInfoFunc(fi, "glGetMapParameterfvNV", glGetMapParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetMapParameterivNV", glGetMapParameterivNV == NULL);
  glewInfoFunc(fi, "glMapControlPointsNV", glMapControlPointsNV == NULL);
  glewInfoFunc(fi, "glMapParameterfvNV", glMapParameterfvNV == NULL);
  glewInfoFunc(fi, "glMapParameterivNV", glMapParameterivNV == NULL);
}

#endif /* GL_NV_evaluators */

#ifdef GL_NV_explicit_attrib_location

static void _glewInfo_GL_NV_explicit_attrib_location (void)
{
  glewPrintExt("GL_NV_explicit_attrib_location", GLEW_NV_explicit_attrib_location, glewIsSupported("GL_NV_explicit_attrib_location"), glewGetExtension("GL_NV_explicit_attrib_location"));
}

#endif /* GL_NV_explicit_attrib_location */

#ifdef GL_NV_explicit_multisample

static void _glewInfo_GL_NV_explicit_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_NV_explicit_multisample", GLEW_NV_explicit_multisample, glewIsSupported("GL_NV_explicit_multisample"), glewGetExtension("GL_NV_explicit_multisample"));

  glewInfoFunc(fi, "glGetMultisamplefvNV", glGetMultisamplefvNV == NULL);
  glewInfoFunc(fi, "glSampleMaskIndexedNV", glSampleMaskIndexedNV == NULL);
  glewInfoFunc(fi, "glTexRenderbufferNV", glTexRenderbufferNV == NULL);
}

#endif /* GL_NV_explicit_multisample */

#ifdef GL_NV_fbo_color_attachments

static void _glewInfo_GL_NV_fbo_color_attachments (void)
{
  glewPrintExt("GL_NV_fbo_color_attachments", GLEW_NV_fbo_color_attachments, glewIsSupported("GL_NV_fbo_color_attachments"), glewGetExtension("GL_NV_fbo_color_attachments"));
}

#endif /* GL_NV_fbo_color_attachments */

#ifdef GL_NV_fence

static void _glewInfo_GL_NV_fence (void)
{
  GLboolean fi = glewPrintExt("GL_NV_fence", GLEW_NV_fence, glewIsSupported("GL_NV_fence"), glewGetExtension("GL_NV_fence"));

  glewInfoFunc(fi, "glDeleteFencesNV", glDeleteFencesNV == NULL);
  glewInfoFunc(fi, "glFinishFenceNV", glFinishFenceNV == NULL);
  glewInfoFunc(fi, "glGenFencesNV", glGenFencesNV == NULL);
  glewInfoFunc(fi, "glGetFenceivNV", glGetFenceivNV == NULL);
  glewInfoFunc(fi, "glIsFenceNV", glIsFenceNV == NULL);
  glewInfoFunc(fi, "glSetFenceNV", glSetFenceNV == NULL);
  glewInfoFunc(fi, "glTestFenceNV", glTestFenceNV == NULL);
}

#endif /* GL_NV_fence */

#ifdef GL_NV_fill_rectangle

static void _glewInfo_GL_NV_fill_rectangle (void)
{
  glewPrintExt("GL_NV_fill_rectangle", GLEW_NV_fill_rectangle, glewIsSupported("GL_NV_fill_rectangle"), glewGetExtension("GL_NV_fill_rectangle"));
}

#endif /* GL_NV_fill_rectangle */

#ifdef GL_NV_float_buffer

static void _glewInfo_GL_NV_float_buffer (void)
{
  glewPrintExt("GL_NV_float_buffer", GLEW_NV_float_buffer, glewIsSupported("GL_NV_float_buffer"), glewGetExtension("GL_NV_float_buffer"));
}

#endif /* GL_NV_float_buffer */

#ifdef GL_NV_fog_distance

static void _glewInfo_GL_NV_fog_distance (void)
{
  glewPrintExt("GL_NV_fog_distance", GLEW_NV_fog_distance, glewIsSupported("GL_NV_fog_distance"), glewGetExtension("GL_NV_fog_distance"));
}

#endif /* GL_NV_fog_distance */

#ifdef GL_NV_fragment_coverage_to_color

static void _glewInfo_GL_NV_fragment_coverage_to_color (void)
{
  GLboolean fi = glewPrintExt("GL_NV_fragment_coverage_to_color", GLEW_NV_fragment_coverage_to_color, glewIsSupported("GL_NV_fragment_coverage_to_color"), glewGetExtension("GL_NV_fragment_coverage_to_color"));

  glewInfoFunc(fi, "glFragmentCoverageColorNV", glFragmentCoverageColorNV == NULL);
}

#endif /* GL_NV_fragment_coverage_to_color */

#ifdef GL_NV_fragment_program

static void _glewInfo_GL_NV_fragment_program (void)
{
  GLboolean fi = glewPrintExt("GL_NV_fragment_program", GLEW_NV_fragment_program, glewIsSupported("GL_NV_fragment_program"), glewGetExtension("GL_NV_fragment_program"));

  glewInfoFunc(fi, "glGetProgramNamedParameterdvNV", glGetProgramNamedParameterdvNV == NULL);
  glewInfoFunc(fi, "glGetProgramNamedParameterfvNV", glGetProgramNamedParameterfvNV == NULL);
  glewInfoFunc(fi, "glProgramNamedParameter4dNV", glProgramNamedParameter4dNV == NULL);
  glewInfoFunc(fi, "glProgramNamedParameter4dvNV", glProgramNamedParameter4dvNV == NULL);
  glewInfoFunc(fi, "glProgramNamedParameter4fNV", glProgramNamedParameter4fNV == NULL);
  glewInfoFunc(fi, "glProgramNamedParameter4fvNV", glProgramNamedParameter4fvNV == NULL);
}

#endif /* GL_NV_fragment_program */

#ifdef GL_NV_fragment_program2

static void _glewInfo_GL_NV_fragment_program2 (void)
{
  glewPrintExt("GL_NV_fragment_program2", GLEW_NV_fragment_program2, glewIsSupported("GL_NV_fragment_program2"), glewGetExtension("GL_NV_fragment_program2"));
}

#endif /* GL_NV_fragment_program2 */

#ifdef GL_NV_fragment_program4

static void _glewInfo_GL_NV_fragment_program4 (void)
{
  glewPrintExt("GL_NV_fragment_program4", GLEW_NV_fragment_program4, glewIsSupported("GL_NV_fragment_program4"), glewGetExtension("GL_NV_gpu_program4"));
}

#endif /* GL_NV_fragment_program4 */

#ifdef GL_NV_fragment_program_option

static void _glewInfo_GL_NV_fragment_program_option (void)
{
  glewPrintExt("GL_NV_fragment_program_option", GLEW_NV_fragment_program_option, glewIsSupported("GL_NV_fragment_program_option"), glewGetExtension("GL_NV_fragment_program_option"));
}

#endif /* GL_NV_fragment_program_option */

#ifdef GL_NV_fragment_shader_barycentric

static void _glewInfo_GL_NV_fragment_shader_barycentric (void)
{
  glewPrintExt("GL_NV_fragment_shader_barycentric", GLEW_NV_fragment_shader_barycentric, glewIsSupported("GL_NV_fragment_shader_barycentric"), glewGetExtension("GL_NV_fragment_shader_barycentric"));
}

#endif /* GL_NV_fragment_shader_barycentric */

#ifdef GL_NV_fragment_shader_interlock

static void _glewInfo_GL_NV_fragment_shader_interlock (void)
{
  glewPrintExt("GL_NV_fragment_shader_interlock", GLEW_NV_fragment_shader_interlock, glewIsSupported("GL_NV_fragment_shader_interlock"), glewGetExtension("GL_NV_fragment_shader_interlock"));
}

#endif /* GL_NV_fragment_shader_interlock */

#ifdef GL_NV_framebuffer_blit

static void _glewInfo_GL_NV_framebuffer_blit (void)
{
  GLboolean fi = glewPrintExt("GL_NV_framebuffer_blit", GLEW_NV_framebuffer_blit, glewIsSupported("GL_NV_framebuffer_blit"), glewGetExtension("GL_NV_framebuffer_blit"));

  glewInfoFunc(fi, "glBlitFramebufferNV", glBlitFramebufferNV == NULL);
}

#endif /* GL_NV_framebuffer_blit */

#ifdef GL_NV_framebuffer_mixed_samples

static void _glewInfo_GL_NV_framebuffer_mixed_samples (void)
{
  glewPrintExt("GL_NV_framebuffer_mixed_samples", GLEW_NV_framebuffer_mixed_samples, glewIsSupported("GL_NV_framebuffer_mixed_samples"), glewGetExtension("GL_NV_framebuffer_mixed_samples"));
}

#endif /* GL_NV_framebuffer_mixed_samples */

#ifdef GL_NV_framebuffer_multisample

static void _glewInfo_GL_NV_framebuffer_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_NV_framebuffer_multisample", GLEW_NV_framebuffer_multisample, glewIsSupported("GL_NV_framebuffer_multisample"), glewGetExtension("GL_NV_framebuffer_multisample"));

  glewInfoFunc(fi, "glRenderbufferStorageMultisampleNV", glRenderbufferStorageMultisampleNV == NULL);
}

#endif /* GL_NV_framebuffer_multisample */

#ifdef GL_NV_framebuffer_multisample_coverage

static void _glewInfo_GL_NV_framebuffer_multisample_coverage (void)
{
  GLboolean fi = glewPrintExt("GL_NV_framebuffer_multisample_coverage", GLEW_NV_framebuffer_multisample_coverage, glewIsSupported("GL_NV_framebuffer_multisample_coverage"), glewGetExtension("GL_NV_framebuffer_multisample_coverage"));

  glewInfoFunc(fi, "glRenderbufferStorageMultisampleCoverageNV", glRenderbufferStorageMultisampleCoverageNV == NULL);
}

#endif /* GL_NV_framebuffer_multisample_coverage */

#ifdef GL_NV_generate_mipmap_sRGB

static void _glewInfo_GL_NV_generate_mipmap_sRGB (void)
{
  glewPrintExt("GL_NV_generate_mipmap_sRGB", GLEW_NV_generate_mipmap_sRGB, glewIsSupported("GL_NV_generate_mipmap_sRGB"), glewGetExtension("GL_NV_generate_mipmap_sRGB"));
}

#endif /* GL_NV_generate_mipmap_sRGB */

#ifdef GL_NV_geometry_program4

static void _glewInfo_GL_NV_geometry_program4 (void)
{
  GLboolean fi = glewPrintExt("GL_NV_geometry_program4", GLEW_NV_geometry_program4, glewIsSupported("GL_NV_geometry_program4"), glewGetExtension("GL_NV_gpu_program4"));

  glewInfoFunc(fi, "glProgramVertexLimitNV", glProgramVertexLimitNV == NULL);
}

#endif /* GL_NV_geometry_program4 */

#ifdef GL_NV_geometry_shader4

static void _glewInfo_GL_NV_geometry_shader4 (void)
{
  glewPrintExt("GL_NV_geometry_shader4", GLEW_NV_geometry_shader4, glewIsSupported("GL_NV_geometry_shader4"), glewGetExtension("GL_NV_geometry_shader4"));
}

#endif /* GL_NV_geometry_shader4 */

#ifdef GL_NV_geometry_shader_passthrough

static void _glewInfo_GL_NV_geometry_shader_passthrough (void)
{
  glewPrintExt("GL_NV_geometry_shader_passthrough", GLEW_NV_geometry_shader_passthrough, glewIsSupported("GL_NV_geometry_shader_passthrough"), glewGetExtension("GL_NV_geometry_shader_passthrough"));
}

#endif /* GL_NV_geometry_shader_passthrough */

#ifdef GL_NV_gpu_multicast

static void _glewInfo_GL_NV_gpu_multicast (void)
{
  GLboolean fi = glewPrintExt("GL_NV_gpu_multicast", GLEW_NV_gpu_multicast, glewIsSupported("GL_NV_gpu_multicast"), glewGetExtension("GL_NV_gpu_multicast"));

  glewInfoFunc(fi, "glMulticastBarrierNV", glMulticastBarrierNV == NULL);
  glewInfoFunc(fi, "glMulticastBlitFramebufferNV", glMulticastBlitFramebufferNV == NULL);
  glewInfoFunc(fi, "glMulticastBufferSubDataNV", glMulticastBufferSubDataNV == NULL);
  glewInfoFunc(fi, "glMulticastCopyBufferSubDataNV", glMulticastCopyBufferSubDataNV == NULL);
  glewInfoFunc(fi, "glMulticastCopyImageSubDataNV", glMulticastCopyImageSubDataNV == NULL);
  glewInfoFunc(fi, "glMulticastFramebufferSampleLocationsfvNV", glMulticastFramebufferSampleLocationsfvNV == NULL);
  glewInfoFunc(fi, "glMulticastGetQueryObjecti64vNV", glMulticastGetQueryObjecti64vNV == NULL);
  glewInfoFunc(fi, "glMulticastGetQueryObjectivNV", glMulticastGetQueryObjectivNV == NULL);
  glewInfoFunc(fi, "glMulticastGetQueryObjectui64vNV", glMulticastGetQueryObjectui64vNV == NULL);
  glewInfoFunc(fi, "glMulticastGetQueryObjectuivNV", glMulticastGetQueryObjectuivNV == NULL);
  glewInfoFunc(fi, "glMulticastWaitSyncNV", glMulticastWaitSyncNV == NULL);
  glewInfoFunc(fi, "glRenderGpuMaskNV", glRenderGpuMaskNV == NULL);
}

#endif /* GL_NV_gpu_multicast */

#ifdef GL_NV_gpu_program4

static void _glewInfo_GL_NV_gpu_program4 (void)
{
  GLboolean fi = glewPrintExt("GL_NV_gpu_program4", GLEW_NV_gpu_program4, glewIsSupported("GL_NV_gpu_program4"), glewGetExtension("GL_NV_gpu_program4"));

  glewInfoFunc(fi, "glProgramEnvParameterI4iNV", glProgramEnvParameterI4iNV == NULL);
  glewInfoFunc(fi, "glProgramEnvParameterI4ivNV", glProgramEnvParameterI4ivNV == NULL);
  glewInfoFunc(fi, "glProgramEnvParameterI4uiNV", glProgramEnvParameterI4uiNV == NULL);
  glewInfoFunc(fi, "glProgramEnvParameterI4uivNV", glProgramEnvParameterI4uivNV == NULL);
  glewInfoFunc(fi, "glProgramEnvParametersI4ivNV", glProgramEnvParametersI4ivNV == NULL);
  glewInfoFunc(fi, "glProgramEnvParametersI4uivNV", glProgramEnvParametersI4uivNV == NULL);
  glewInfoFunc(fi, "glProgramLocalParameterI4iNV", glProgramLocalParameterI4iNV == NULL);
  glewInfoFunc(fi, "glProgramLocalParameterI4ivNV", glProgramLocalParameterI4ivNV == NULL);
  glewInfoFunc(fi, "glProgramLocalParameterI4uiNV", glProgramLocalParameterI4uiNV == NULL);
  glewInfoFunc(fi, "glProgramLocalParameterI4uivNV", glProgramLocalParameterI4uivNV == NULL);
  glewInfoFunc(fi, "glProgramLocalParametersI4ivNV", glProgramLocalParametersI4ivNV == NULL);
  glewInfoFunc(fi, "glProgramLocalParametersI4uivNV", glProgramLocalParametersI4uivNV == NULL);
}

#endif /* GL_NV_gpu_program4 */

#ifdef GL_NV_gpu_program5

static void _glewInfo_GL_NV_gpu_program5 (void)
{
  glewPrintExt("GL_NV_gpu_program5", GLEW_NV_gpu_program5, glewIsSupported("GL_NV_gpu_program5"), glewGetExtension("GL_NV_gpu_program5"));
}

#endif /* GL_NV_gpu_program5 */

#ifdef GL_NV_gpu_program5_mem_extended

static void _glewInfo_GL_NV_gpu_program5_mem_extended (void)
{
  glewPrintExt("GL_NV_gpu_program5_mem_extended", GLEW_NV_gpu_program5_mem_extended, glewIsSupported("GL_NV_gpu_program5_mem_extended"), glewGetExtension("GL_NV_gpu_program5_mem_extended"));
}

#endif /* GL_NV_gpu_program5_mem_extended */

#ifdef GL_NV_gpu_program_fp64

static void _glewInfo_GL_NV_gpu_program_fp64 (void)
{
  glewPrintExt("GL_NV_gpu_program_fp64", GLEW_NV_gpu_program_fp64, glewIsSupported("GL_NV_gpu_program_fp64"), glewGetExtension("GL_NV_gpu_program_fp64"));
}

#endif /* GL_NV_gpu_program_fp64 */

#ifdef GL_NV_gpu_shader5

static void _glewInfo_GL_NV_gpu_shader5 (void)
{
  GLboolean fi = glewPrintExt("GL_NV_gpu_shader5", GLEW_NV_gpu_shader5, glewIsSupported("GL_NV_gpu_shader5"), glewGetExtension("GL_NV_gpu_shader5"));

  glewInfoFunc(fi, "glGetUniformi64vNV", glGetUniformi64vNV == NULL);
  glewInfoFunc(fi, "glGetUniformui64vNV", glGetUniformui64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform1i64NV", glProgramUniform1i64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform1i64vNV", glProgramUniform1i64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform1ui64NV", glProgramUniform1ui64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform1ui64vNV", glProgramUniform1ui64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform2i64NV", glProgramUniform2i64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform2i64vNV", glProgramUniform2i64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform2ui64NV", glProgramUniform2ui64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform2ui64vNV", glProgramUniform2ui64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform3i64NV", glProgramUniform3i64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform3i64vNV", glProgramUniform3i64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform3ui64NV", glProgramUniform3ui64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform3ui64vNV", glProgramUniform3ui64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform4i64NV", glProgramUniform4i64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform4i64vNV", glProgramUniform4i64vNV == NULL);
  glewInfoFunc(fi, "glProgramUniform4ui64NV", glProgramUniform4ui64NV == NULL);
  glewInfoFunc(fi, "glProgramUniform4ui64vNV", glProgramUniform4ui64vNV == NULL);
  glewInfoFunc(fi, "glUniform1i64NV", glUniform1i64NV == NULL);
  glewInfoFunc(fi, "glUniform1i64vNV", glUniform1i64vNV == NULL);
  glewInfoFunc(fi, "glUniform1ui64NV", glUniform1ui64NV == NULL);
  glewInfoFunc(fi, "glUniform1ui64vNV", glUniform1ui64vNV == NULL);
  glewInfoFunc(fi, "glUniform2i64NV", glUniform2i64NV == NULL);
  glewInfoFunc(fi, "glUniform2i64vNV", glUniform2i64vNV == NULL);
  glewInfoFunc(fi, "glUniform2ui64NV", glUniform2ui64NV == NULL);
  glewInfoFunc(fi, "glUniform2ui64vNV", glUniform2ui64vNV == NULL);
  glewInfoFunc(fi, "glUniform3i64NV", glUniform3i64NV == NULL);
  glewInfoFunc(fi, "glUniform3i64vNV", glUniform3i64vNV == NULL);
  glewInfoFunc(fi, "glUniform3ui64NV", glUniform3ui64NV == NULL);
  glewInfoFunc(fi, "glUniform3ui64vNV", glUniform3ui64vNV == NULL);
  glewInfoFunc(fi, "glUniform4i64NV", glUniform4i64NV == NULL);
  glewInfoFunc(fi, "glUniform4i64vNV", glUniform4i64vNV == NULL);
  glewInfoFunc(fi, "glUniform4ui64NV", glUniform4ui64NV == NULL);
  glewInfoFunc(fi, "glUniform4ui64vNV", glUniform4ui64vNV == NULL);
}

#endif /* GL_NV_gpu_shader5 */

#ifdef GL_NV_half_float

static void _glewInfo_GL_NV_half_float (void)
{
  GLboolean fi = glewPrintExt("GL_NV_half_float", GLEW_NV_half_float, glewIsSupported("GL_NV_half_float"), glewGetExtension("GL_NV_half_float"));

  glewInfoFunc(fi, "glColor3hNV", glColor3hNV == NULL);
  glewInfoFunc(fi, "glColor3hvNV", glColor3hvNV == NULL);
  glewInfoFunc(fi, "glColor4hNV", glColor4hNV == NULL);
  glewInfoFunc(fi, "glColor4hvNV", glColor4hvNV == NULL);
  glewInfoFunc(fi, "glFogCoordhNV", glFogCoordhNV == NULL);
  glewInfoFunc(fi, "glFogCoordhvNV", glFogCoordhvNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1hNV", glMultiTexCoord1hNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord1hvNV", glMultiTexCoord1hvNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2hNV", glMultiTexCoord2hNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord2hvNV", glMultiTexCoord2hvNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3hNV", glMultiTexCoord3hNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord3hvNV", glMultiTexCoord3hvNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4hNV", glMultiTexCoord4hNV == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4hvNV", glMultiTexCoord4hvNV == NULL);
  glewInfoFunc(fi, "glNormal3hNV", glNormal3hNV == NULL);
  glewInfoFunc(fi, "glNormal3hvNV", glNormal3hvNV == NULL);
  glewInfoFunc(fi, "glSecondaryColor3hNV", glSecondaryColor3hNV == NULL);
  glewInfoFunc(fi, "glSecondaryColor3hvNV", glSecondaryColor3hvNV == NULL);
  glewInfoFunc(fi, "glTexCoord1hNV", glTexCoord1hNV == NULL);
  glewInfoFunc(fi, "glTexCoord1hvNV", glTexCoord1hvNV == NULL);
  glewInfoFunc(fi, "glTexCoord2hNV", glTexCoord2hNV == NULL);
  glewInfoFunc(fi, "glTexCoord2hvNV", glTexCoord2hvNV == NULL);
  glewInfoFunc(fi, "glTexCoord3hNV", glTexCoord3hNV == NULL);
  glewInfoFunc(fi, "glTexCoord3hvNV", glTexCoord3hvNV == NULL);
  glewInfoFunc(fi, "glTexCoord4hNV", glTexCoord4hNV == NULL);
  glewInfoFunc(fi, "glTexCoord4hvNV", glTexCoord4hvNV == NULL);
  glewInfoFunc(fi, "glVertex2hNV", glVertex2hNV == NULL);
  glewInfoFunc(fi, "glVertex2hvNV", glVertex2hvNV == NULL);
  glewInfoFunc(fi, "glVertex3hNV", glVertex3hNV == NULL);
  glewInfoFunc(fi, "glVertex3hvNV", glVertex3hvNV == NULL);
  glewInfoFunc(fi, "glVertex4hNV", glVertex4hNV == NULL);
  glewInfoFunc(fi, "glVertex4hvNV", glVertex4hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1hNV", glVertexAttrib1hNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1hvNV", glVertexAttrib1hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2hNV", glVertexAttrib2hNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2hvNV", glVertexAttrib2hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3hNV", glVertexAttrib3hNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3hvNV", glVertexAttrib3hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4hNV", glVertexAttrib4hNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4hvNV", glVertexAttrib4hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs1hvNV", glVertexAttribs1hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs2hvNV", glVertexAttribs2hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs3hvNV", glVertexAttribs3hvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs4hvNV", glVertexAttribs4hvNV == NULL);
  glewInfoFunc(fi, "glVertexWeighthNV", glVertexWeighthNV == NULL);
  glewInfoFunc(fi, "glVertexWeighthvNV", glVertexWeighthvNV == NULL);
}

#endif /* GL_NV_half_float */

#ifdef GL_NV_image_formats

static void _glewInfo_GL_NV_image_formats (void)
{
  glewPrintExt("GL_NV_image_formats", GLEW_NV_image_formats, glewIsSupported("GL_NV_image_formats"), glewGetExtension("GL_NV_image_formats"));
}

#endif /* GL_NV_image_formats */

#ifdef GL_NV_instanced_arrays

static void _glewInfo_GL_NV_instanced_arrays (void)
{
  GLboolean fi = glewPrintExt("GL_NV_instanced_arrays", GLEW_NV_instanced_arrays, glewIsSupported("GL_NV_instanced_arrays"), glewGetExtension("GL_NV_instanced_arrays"));

  glewInfoFunc(fi, "glVertexAttribDivisorNV", glVertexAttribDivisorNV == NULL);
}

#endif /* GL_NV_instanced_arrays */

#ifdef GL_NV_internalformat_sample_query

static void _glewInfo_GL_NV_internalformat_sample_query (void)
{
  GLboolean fi = glewPrintExt("GL_NV_internalformat_sample_query", GLEW_NV_internalformat_sample_query, glewIsSupported("GL_NV_internalformat_sample_query"), glewGetExtension("GL_NV_internalformat_sample_query"));

  glewInfoFunc(fi, "glGetInternalformatSampleivNV", glGetInternalformatSampleivNV == NULL);
}

#endif /* GL_NV_internalformat_sample_query */

#ifdef GL_NV_light_max_exponent

static void _glewInfo_GL_NV_light_max_exponent (void)
{
  glewPrintExt("GL_NV_light_max_exponent", GLEW_NV_light_max_exponent, glewIsSupported("GL_NV_light_max_exponent"), glewGetExtension("GL_NV_light_max_exponent"));
}

#endif /* GL_NV_light_max_exponent */

#ifdef GL_NV_memory_attachment

static void _glewInfo_GL_NV_memory_attachment (void)
{
  GLboolean fi = glewPrintExt("GL_NV_memory_attachment", GLEW_NV_memory_attachment, glewIsSupported("GL_NV_memory_attachment"), glewGetExtension("GL_NV_memory_attachment"));

  glewInfoFunc(fi, "glBufferAttachMemoryNV", glBufferAttachMemoryNV == NULL);
  glewInfoFunc(fi, "glGetMemoryObjectDetachedResourcesuivNV", glGetMemoryObjectDetachedResourcesuivNV == NULL);
  glewInfoFunc(fi, "glNamedBufferAttachMemoryNV", glNamedBufferAttachMemoryNV == NULL);
  glewInfoFunc(fi, "glResetMemoryObjectParameterNV", glResetMemoryObjectParameterNV == NULL);
  glewInfoFunc(fi, "glTexAttachMemoryNV", glTexAttachMemoryNV == NULL);
  glewInfoFunc(fi, "glTextureAttachMemoryNV", glTextureAttachMemoryNV == NULL);
}

#endif /* GL_NV_memory_attachment */

#ifdef GL_NV_mesh_shader

static void _glewInfo_GL_NV_mesh_shader (void)
{
  GLboolean fi = glewPrintExt("GL_NV_mesh_shader", GLEW_NV_mesh_shader, glewIsSupported("GL_NV_mesh_shader"), glewGetExtension("GL_NV_mesh_shader"));

  glewInfoFunc(fi, "glDrawMeshTasksIndirectNV", glDrawMeshTasksIndirectNV == NULL);
  glewInfoFunc(fi, "glDrawMeshTasksNV", glDrawMeshTasksNV == NULL);
  glewInfoFunc(fi, "glMultiDrawMeshTasksIndirectCountNV", glMultiDrawMeshTasksIndirectCountNV == NULL);
  glewInfoFunc(fi, "glMultiDrawMeshTasksIndirectNV", glMultiDrawMeshTasksIndirectNV == NULL);
}

#endif /* GL_NV_mesh_shader */

#ifdef GL_NV_multisample_coverage

static void _glewInfo_GL_NV_multisample_coverage (void)
{
  glewPrintExt("GL_NV_multisample_coverage", GLEW_NV_multisample_coverage, glewIsSupported("GL_NV_multisample_coverage"), glewGetExtension("GL_NV_multisample_coverage"));
}

#endif /* GL_NV_multisample_coverage */

#ifdef GL_NV_multisample_filter_hint

static void _glewInfo_GL_NV_multisample_filter_hint (void)
{
  glewPrintExt("GL_NV_multisample_filter_hint", GLEW_NV_multisample_filter_hint, glewIsSupported("GL_NV_multisample_filter_hint"), glewGetExtension("GL_NV_multisample_filter_hint"));
}

#endif /* GL_NV_multisample_filter_hint */

#ifdef GL_NV_non_square_matrices

static void _glewInfo_GL_NV_non_square_matrices (void)
{
  GLboolean fi = glewPrintExt("GL_NV_non_square_matrices", GLEW_NV_non_square_matrices, glewIsSupported("GL_NV_non_square_matrices"), glewGetExtension("GL_NV_non_square_matrices"));

  glewInfoFunc(fi, "glUniformMatrix2x3fvNV", glUniformMatrix2x3fvNV == NULL);
  glewInfoFunc(fi, "glUniformMatrix2x4fvNV", glUniformMatrix2x4fvNV == NULL);
  glewInfoFunc(fi, "glUniformMatrix3x2fvNV", glUniformMatrix3x2fvNV == NULL);
  glewInfoFunc(fi, "glUniformMatrix3x4fvNV", glUniformMatrix3x4fvNV == NULL);
  glewInfoFunc(fi, "glUniformMatrix4x2fvNV", glUniformMatrix4x2fvNV == NULL);
  glewInfoFunc(fi, "glUniformMatrix4x3fvNV", glUniformMatrix4x3fvNV == NULL);
}

#endif /* GL_NV_non_square_matrices */

#ifdef GL_NV_occlusion_query

static void _glewInfo_GL_NV_occlusion_query (void)
{
  GLboolean fi = glewPrintExt("GL_NV_occlusion_query", GLEW_NV_occlusion_query, glewIsSupported("GL_NV_occlusion_query"), glewGetExtension("GL_NV_occlusion_query"));

  glewInfoFunc(fi, "glBeginOcclusionQueryNV", glBeginOcclusionQueryNV == NULL);
  glewInfoFunc(fi, "glDeleteOcclusionQueriesNV", glDeleteOcclusionQueriesNV == NULL);
  glewInfoFunc(fi, "glEndOcclusionQueryNV", glEndOcclusionQueryNV == NULL);
  glewInfoFunc(fi, "glGenOcclusionQueriesNV", glGenOcclusionQueriesNV == NULL);
  glewInfoFunc(fi, "glGetOcclusionQueryivNV", glGetOcclusionQueryivNV == NULL);
  glewInfoFunc(fi, "glGetOcclusionQueryuivNV", glGetOcclusionQueryuivNV == NULL);
  glewInfoFunc(fi, "glIsOcclusionQueryNV", glIsOcclusionQueryNV == NULL);
}

#endif /* GL_NV_occlusion_query */

#ifdef GL_NV_pack_subimage

static void _glewInfo_GL_NV_pack_subimage (void)
{
  glewPrintExt("GL_NV_pack_subimage", GLEW_NV_pack_subimage, glewIsSupported("GL_NV_pack_subimage"), glewGetExtension("GL_NV_pack_subimage"));
}

#endif /* GL_NV_pack_subimage */

#ifdef GL_NV_packed_depth_stencil

static void _glewInfo_GL_NV_packed_depth_stencil (void)
{
  glewPrintExt("GL_NV_packed_depth_stencil", GLEW_NV_packed_depth_stencil, glewIsSupported("GL_NV_packed_depth_stencil"), glewGetExtension("GL_NV_packed_depth_stencil"));
}

#endif /* GL_NV_packed_depth_stencil */

#ifdef GL_NV_packed_float

static void _glewInfo_GL_NV_packed_float (void)
{
  glewPrintExt("GL_NV_packed_float", GLEW_NV_packed_float, glewIsSupported("GL_NV_packed_float"), glewGetExtension("GL_NV_packed_float"));
}

#endif /* GL_NV_packed_float */

#ifdef GL_NV_packed_float_linear

static void _glewInfo_GL_NV_packed_float_linear (void)
{
  glewPrintExt("GL_NV_packed_float_linear", GLEW_NV_packed_float_linear, glewIsSupported("GL_NV_packed_float_linear"), glewGetExtension("GL_NV_packed_float_linear"));
}

#endif /* GL_NV_packed_float_linear */

#ifdef GL_NV_parameter_buffer_object

static void _glewInfo_GL_NV_parameter_buffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_NV_parameter_buffer_object", GLEW_NV_parameter_buffer_object, glewIsSupported("GL_NV_parameter_buffer_object"), glewGetExtension("GL_NV_parameter_buffer_object"));

  glewInfoFunc(fi, "glProgramBufferParametersIivNV", glProgramBufferParametersIivNV == NULL);
  glewInfoFunc(fi, "glProgramBufferParametersIuivNV", glProgramBufferParametersIuivNV == NULL);
  glewInfoFunc(fi, "glProgramBufferParametersfvNV", glProgramBufferParametersfvNV == NULL);
}

#endif /* GL_NV_parameter_buffer_object */

#ifdef GL_NV_parameter_buffer_object2

static void _glewInfo_GL_NV_parameter_buffer_object2 (void)
{
  glewPrintExt("GL_NV_parameter_buffer_object2", GLEW_NV_parameter_buffer_object2, glewIsSupported("GL_NV_parameter_buffer_object2"), glewGetExtension("GL_NV_parameter_buffer_object2"));
}

#endif /* GL_NV_parameter_buffer_object2 */

#ifdef GL_NV_path_rendering

static void _glewInfo_GL_NV_path_rendering (void)
{
  GLboolean fi = glewPrintExt("GL_NV_path_rendering", GLEW_NV_path_rendering, glewIsSupported("GL_NV_path_rendering"), glewGetExtension("GL_NV_path_rendering"));

  glewInfoFunc(fi, "glCopyPathNV", glCopyPathNV == NULL);
  glewInfoFunc(fi, "glCoverFillPathInstancedNV", glCoverFillPathInstancedNV == NULL);
  glewInfoFunc(fi, "glCoverFillPathNV", glCoverFillPathNV == NULL);
  glewInfoFunc(fi, "glCoverStrokePathInstancedNV", glCoverStrokePathInstancedNV == NULL);
  glewInfoFunc(fi, "glCoverStrokePathNV", glCoverStrokePathNV == NULL);
  glewInfoFunc(fi, "glDeletePathsNV", glDeletePathsNV == NULL);
  glewInfoFunc(fi, "glGenPathsNV", glGenPathsNV == NULL);
  glewInfoFunc(fi, "glGetPathColorGenfvNV", glGetPathColorGenfvNV == NULL);
  glewInfoFunc(fi, "glGetPathColorGenivNV", glGetPathColorGenivNV == NULL);
  glewInfoFunc(fi, "glGetPathCommandsNV", glGetPathCommandsNV == NULL);
  glewInfoFunc(fi, "glGetPathCoordsNV", glGetPathCoordsNV == NULL);
  glewInfoFunc(fi, "glGetPathDashArrayNV", glGetPathDashArrayNV == NULL);
  glewInfoFunc(fi, "glGetPathLengthNV", glGetPathLengthNV == NULL);
  glewInfoFunc(fi, "glGetPathMetricRangeNV", glGetPathMetricRangeNV == NULL);
  glewInfoFunc(fi, "glGetPathMetricsNV", glGetPathMetricsNV == NULL);
  glewInfoFunc(fi, "glGetPathParameterfvNV", glGetPathParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetPathParameterivNV", glGetPathParameterivNV == NULL);
  glewInfoFunc(fi, "glGetPathSpacingNV", glGetPathSpacingNV == NULL);
  glewInfoFunc(fi, "glGetPathTexGenfvNV", glGetPathTexGenfvNV == NULL);
  glewInfoFunc(fi, "glGetPathTexGenivNV", glGetPathTexGenivNV == NULL);
  glewInfoFunc(fi, "glGetProgramResourcefvNV", glGetProgramResourcefvNV == NULL);
  glewInfoFunc(fi, "glInterpolatePathsNV", glInterpolatePathsNV == NULL);
  glewInfoFunc(fi, "glIsPathNV", glIsPathNV == NULL);
  glewInfoFunc(fi, "glIsPointInFillPathNV", glIsPointInFillPathNV == NULL);
  glewInfoFunc(fi, "glIsPointInStrokePathNV", glIsPointInStrokePathNV == NULL);
  glewInfoFunc(fi, "glMatrixLoad3x2fNV", glMatrixLoad3x2fNV == NULL);
  glewInfoFunc(fi, "glMatrixLoad3x3fNV", glMatrixLoad3x3fNV == NULL);
  glewInfoFunc(fi, "glMatrixLoadTranspose3x3fNV", glMatrixLoadTranspose3x3fNV == NULL);
  glewInfoFunc(fi, "glMatrixMult3x2fNV", glMatrixMult3x2fNV == NULL);
  glewInfoFunc(fi, "glMatrixMult3x3fNV", glMatrixMult3x3fNV == NULL);
  glewInfoFunc(fi, "glMatrixMultTranspose3x3fNV", glMatrixMultTranspose3x3fNV == NULL);
  glewInfoFunc(fi, "glPathColorGenNV", glPathColorGenNV == NULL);
  glewInfoFunc(fi, "glPathCommandsNV", glPathCommandsNV == NULL);
  glewInfoFunc(fi, "glPathCoordsNV", glPathCoordsNV == NULL);
  glewInfoFunc(fi, "glPathCoverDepthFuncNV", glPathCoverDepthFuncNV == NULL);
  glewInfoFunc(fi, "glPathDashArrayNV", glPathDashArrayNV == NULL);
  glewInfoFunc(fi, "glPathFogGenNV", glPathFogGenNV == NULL);
  glewInfoFunc(fi, "glPathGlyphIndexArrayNV", glPathGlyphIndexArrayNV == NULL);
  glewInfoFunc(fi, "glPathGlyphIndexRangeNV", glPathGlyphIndexRangeNV == NULL);
  glewInfoFunc(fi, "glPathGlyphRangeNV", glPathGlyphRangeNV == NULL);
  glewInfoFunc(fi, "glPathGlyphsNV", glPathGlyphsNV == NULL);
  glewInfoFunc(fi, "glPathMemoryGlyphIndexArrayNV", glPathMemoryGlyphIndexArrayNV == NULL);
  glewInfoFunc(fi, "glPathParameterfNV", glPathParameterfNV == NULL);
  glewInfoFunc(fi, "glPathParameterfvNV", glPathParameterfvNV == NULL);
  glewInfoFunc(fi, "glPathParameteriNV", glPathParameteriNV == NULL);
  glewInfoFunc(fi, "glPathParameterivNV", glPathParameterivNV == NULL);
  glewInfoFunc(fi, "glPathStencilDepthOffsetNV", glPathStencilDepthOffsetNV == NULL);
  glewInfoFunc(fi, "glPathStencilFuncNV", glPathStencilFuncNV == NULL);
  glewInfoFunc(fi, "glPathStringNV", glPathStringNV == NULL);
  glewInfoFunc(fi, "glPathSubCommandsNV", glPathSubCommandsNV == NULL);
  glewInfoFunc(fi, "glPathSubCoordsNV", glPathSubCoordsNV == NULL);
  glewInfoFunc(fi, "glPathTexGenNV", glPathTexGenNV == NULL);
  glewInfoFunc(fi, "glPointAlongPathNV", glPointAlongPathNV == NULL);
  glewInfoFunc(fi, "glProgramPathFragmentInputGenNV", glProgramPathFragmentInputGenNV == NULL);
  glewInfoFunc(fi, "glStencilFillPathInstancedNV", glStencilFillPathInstancedNV == NULL);
  glewInfoFunc(fi, "glStencilFillPathNV", glStencilFillPathNV == NULL);
  glewInfoFunc(fi, "glStencilStrokePathInstancedNV", glStencilStrokePathInstancedNV == NULL);
  glewInfoFunc(fi, "glStencilStrokePathNV", glStencilStrokePathNV == NULL);
  glewInfoFunc(fi, "glStencilThenCoverFillPathInstancedNV", glStencilThenCoverFillPathInstancedNV == NULL);
  glewInfoFunc(fi, "glStencilThenCoverFillPathNV", glStencilThenCoverFillPathNV == NULL);
  glewInfoFunc(fi, "glStencilThenCoverStrokePathInstancedNV", glStencilThenCoverStrokePathInstancedNV == NULL);
  glewInfoFunc(fi, "glStencilThenCoverStrokePathNV", glStencilThenCoverStrokePathNV == NULL);
  glewInfoFunc(fi, "glTransformPathNV", glTransformPathNV == NULL);
  glewInfoFunc(fi, "glWeightPathsNV", glWeightPathsNV == NULL);
}

#endif /* GL_NV_path_rendering */

#ifdef GL_NV_path_rendering_shared_edge

static void _glewInfo_GL_NV_path_rendering_shared_edge (void)
{
  glewPrintExt("GL_NV_path_rendering_shared_edge", GLEW_NV_path_rendering_shared_edge, glewIsSupported("GL_NV_path_rendering_shared_edge"), glewGetExtension("GL_NV_path_rendering_shared_edge"));
}

#endif /* GL_NV_path_rendering_shared_edge */

#ifdef GL_NV_pixel_buffer_object

static void _glewInfo_GL_NV_pixel_buffer_object (void)
{
  glewPrintExt("GL_NV_pixel_buffer_object", GLEW_NV_pixel_buffer_object, glewIsSupported("GL_NV_pixel_buffer_object"), glewGetExtension("GL_NV_pixel_buffer_object"));
}

#endif /* GL_NV_pixel_buffer_object */

#ifdef GL_NV_pixel_data_range

static void _glewInfo_GL_NV_pixel_data_range (void)
{
  GLboolean fi = glewPrintExt("GL_NV_pixel_data_range", GLEW_NV_pixel_data_range, glewIsSupported("GL_NV_pixel_data_range"), glewGetExtension("GL_NV_pixel_data_range"));

  glewInfoFunc(fi, "glFlushPixelDataRangeNV", glFlushPixelDataRangeNV == NULL);
  glewInfoFunc(fi, "glPixelDataRangeNV", glPixelDataRangeNV == NULL);
}

#endif /* GL_NV_pixel_data_range */

#ifdef GL_NV_platform_binary

static void _glewInfo_GL_NV_platform_binary (void)
{
  glewPrintExt("GL_NV_platform_binary", GLEW_NV_platform_binary, glewIsSupported("GL_NV_platform_binary"), glewGetExtension("GL_NV_platform_binary"));
}

#endif /* GL_NV_platform_binary */

#ifdef GL_NV_point_sprite

static void _glewInfo_GL_NV_point_sprite (void)
{
  GLboolean fi = glewPrintExt("GL_NV_point_sprite", GLEW_NV_point_sprite, glewIsSupported("GL_NV_point_sprite"), glewGetExtension("GL_NV_point_sprite"));

  glewInfoFunc(fi, "glPointParameteriNV", glPointParameteriNV == NULL);
  glewInfoFunc(fi, "glPointParameterivNV", glPointParameterivNV == NULL);
}

#endif /* GL_NV_point_sprite */

#ifdef GL_NV_polygon_mode

static void _glewInfo_GL_NV_polygon_mode (void)
{
  GLboolean fi = glewPrintExt("GL_NV_polygon_mode", GLEW_NV_polygon_mode, glewIsSupported("GL_NV_polygon_mode"), glewGetExtension("GL_NV_polygon_mode"));

  glewInfoFunc(fi, "glPolygonModeNV", glPolygonModeNV == NULL);
}

#endif /* GL_NV_polygon_mode */

#ifdef GL_NV_present_video

static void _glewInfo_GL_NV_present_video (void)
{
  GLboolean fi = glewPrintExt("GL_NV_present_video", GLEW_NV_present_video, glewIsSupported("GL_NV_present_video"), glewGetExtension("GL_NV_present_video"));

  glewInfoFunc(fi, "glGetVideoi64vNV", glGetVideoi64vNV == NULL);
  glewInfoFunc(fi, "glGetVideoivNV", glGetVideoivNV == NULL);
  glewInfoFunc(fi, "glGetVideoui64vNV", glGetVideoui64vNV == NULL);
  glewInfoFunc(fi, "glGetVideouivNV", glGetVideouivNV == NULL);
  glewInfoFunc(fi, "glPresentFrameDualFillNV", glPresentFrameDualFillNV == NULL);
  glewInfoFunc(fi, "glPresentFrameKeyedNV", glPresentFrameKeyedNV == NULL);
}

#endif /* GL_NV_present_video */

#ifdef GL_NV_primitive_restart

static void _glewInfo_GL_NV_primitive_restart (void)
{
  GLboolean fi = glewPrintExt("GL_NV_primitive_restart", GLEW_NV_primitive_restart, glewIsSupported("GL_NV_primitive_restart"), glewGetExtension("GL_NV_primitive_restart"));

  glewInfoFunc(fi, "glPrimitiveRestartIndexNV", glPrimitiveRestartIndexNV == NULL);
  glewInfoFunc(fi, "glPrimitiveRestartNV", glPrimitiveRestartNV == NULL);
}

#endif /* GL_NV_primitive_restart */

#ifdef GL_NV_query_resource_tag

static void _glewInfo_GL_NV_query_resource_tag (void)
{
  glewPrintExt("GL_NV_query_resource_tag", GLEW_NV_query_resource_tag, glewIsSupported("GL_NV_query_resource_tag"), glewGetExtension("GL_NV_query_resource_tag"));
}

#endif /* GL_NV_query_resource_tag */

#ifdef GL_NV_read_buffer

static void _glewInfo_GL_NV_read_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_NV_read_buffer", GLEW_NV_read_buffer, glewIsSupported("GL_NV_read_buffer"), glewGetExtension("GL_NV_read_buffer"));

  glewInfoFunc(fi, "glReadBufferNV", glReadBufferNV == NULL);
}

#endif /* GL_NV_read_buffer */

#ifdef GL_NV_read_buffer_front

static void _glewInfo_GL_NV_read_buffer_front (void)
{
  glewPrintExt("GL_NV_read_buffer_front", GLEW_NV_read_buffer_front, glewIsSupported("GL_NV_read_buffer_front"), glewGetExtension("GL_NV_read_buffer_front"));
}

#endif /* GL_NV_read_buffer_front */

#ifdef GL_NV_read_depth

static void _glewInfo_GL_NV_read_depth (void)
{
  glewPrintExt("GL_NV_read_depth", GLEW_NV_read_depth, glewIsSupported("GL_NV_read_depth"), glewGetExtension("GL_NV_read_depth"));
}

#endif /* GL_NV_read_depth */

#ifdef GL_NV_read_depth_stencil

static void _glewInfo_GL_NV_read_depth_stencil (void)
{
  glewPrintExt("GL_NV_read_depth_stencil", GLEW_NV_read_depth_stencil, glewIsSupported("GL_NV_read_depth_stencil"), glewGetExtension("GL_NV_read_depth_stencil"));
}

#endif /* GL_NV_read_depth_stencil */

#ifdef GL_NV_read_stencil

static void _glewInfo_GL_NV_read_stencil (void)
{
  glewPrintExt("GL_NV_read_stencil", GLEW_NV_read_stencil, glewIsSupported("GL_NV_read_stencil"), glewGetExtension("GL_NV_read_stencil"));
}

#endif /* GL_NV_read_stencil */

#ifdef GL_NV_register_combiners

static void _glewInfo_GL_NV_register_combiners (void)
{
  GLboolean fi = glewPrintExt("GL_NV_register_combiners", GLEW_NV_register_combiners, glewIsSupported("GL_NV_register_combiners"), glewGetExtension("GL_NV_register_combiners"));

  glewInfoFunc(fi, "glCombinerInputNV", glCombinerInputNV == NULL);
  glewInfoFunc(fi, "glCombinerOutputNV", glCombinerOutputNV == NULL);
  glewInfoFunc(fi, "glCombinerParameterfNV", glCombinerParameterfNV == NULL);
  glewInfoFunc(fi, "glCombinerParameterfvNV", glCombinerParameterfvNV == NULL);
  glewInfoFunc(fi, "glCombinerParameteriNV", glCombinerParameteriNV == NULL);
  glewInfoFunc(fi, "glCombinerParameterivNV", glCombinerParameterivNV == NULL);
  glewInfoFunc(fi, "glFinalCombinerInputNV", glFinalCombinerInputNV == NULL);
  glewInfoFunc(fi, "glGetCombinerInputParameterfvNV", glGetCombinerInputParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetCombinerInputParameterivNV", glGetCombinerInputParameterivNV == NULL);
  glewInfoFunc(fi, "glGetCombinerOutputParameterfvNV", glGetCombinerOutputParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetCombinerOutputParameterivNV", glGetCombinerOutputParameterivNV == NULL);
  glewInfoFunc(fi, "glGetFinalCombinerInputParameterfvNV", glGetFinalCombinerInputParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetFinalCombinerInputParameterivNV", glGetFinalCombinerInputParameterivNV == NULL);
}

#endif /* GL_NV_register_combiners */

#ifdef GL_NV_register_combiners2

static void _glewInfo_GL_NV_register_combiners2 (void)
{
  GLboolean fi = glewPrintExt("GL_NV_register_combiners2", GLEW_NV_register_combiners2, glewIsSupported("GL_NV_register_combiners2"), glewGetExtension("GL_NV_register_combiners2"));

  glewInfoFunc(fi, "glCombinerStageParameterfvNV", glCombinerStageParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetCombinerStageParameterfvNV", glGetCombinerStageParameterfvNV == NULL);
}

#endif /* GL_NV_register_combiners2 */

#ifdef GL_NV_representative_fragment_test

static void _glewInfo_GL_NV_representative_fragment_test (void)
{
  glewPrintExt("GL_NV_representative_fragment_test", GLEW_NV_representative_fragment_test, glewIsSupported("GL_NV_representative_fragment_test"), glewGetExtension("GL_NV_representative_fragment_test"));
}

#endif /* GL_NV_representative_fragment_test */

#ifdef GL_NV_robustness_video_memory_purge

static void _glewInfo_GL_NV_robustness_video_memory_purge (void)
{
  glewPrintExt("GL_NV_robustness_video_memory_purge", GLEW_NV_robustness_video_memory_purge, glewIsSupported("GL_NV_robustness_video_memory_purge"), glewGetExtension("GL_NV_robustness_video_memory_purge"));
}

#endif /* GL_NV_robustness_video_memory_purge */

#ifdef GL_NV_sRGB_formats

static void _glewInfo_GL_NV_sRGB_formats (void)
{
  glewPrintExt("GL_NV_sRGB_formats", GLEW_NV_sRGB_formats, glewIsSupported("GL_NV_sRGB_formats"), glewGetExtension("GL_NV_sRGB_formats"));
}

#endif /* GL_NV_sRGB_formats */

#ifdef GL_NV_sample_locations

static void _glewInfo_GL_NV_sample_locations (void)
{
  GLboolean fi = glewPrintExt("GL_NV_sample_locations", GLEW_NV_sample_locations, glewIsSupported("GL_NV_sample_locations"), glewGetExtension("GL_NV_sample_locations"));

  glewInfoFunc(fi, "glFramebufferSampleLocationsfvNV", glFramebufferSampleLocationsfvNV == NULL);
  glewInfoFunc(fi, "glNamedFramebufferSampleLocationsfvNV", glNamedFramebufferSampleLocationsfvNV == NULL);
  glewInfoFunc(fi, "glResolveDepthValuesNV", glResolveDepthValuesNV == NULL);
}

#endif /* GL_NV_sample_locations */

#ifdef GL_NV_sample_mask_override_coverage

static void _glewInfo_GL_NV_sample_mask_override_coverage (void)
{
  glewPrintExt("GL_NV_sample_mask_override_coverage", GLEW_NV_sample_mask_override_coverage, glewIsSupported("GL_NV_sample_mask_override_coverage"), glewGetExtension("GL_NV_sample_mask_override_coverage"));
}

#endif /* GL_NV_sample_mask_override_coverage */

#ifdef GL_NV_scissor_exclusive

static void _glewInfo_GL_NV_scissor_exclusive (void)
{
  GLboolean fi = glewPrintExt("GL_NV_scissor_exclusive", GLEW_NV_scissor_exclusive, glewIsSupported("GL_NV_scissor_exclusive"), glewGetExtension("GL_NV_scissor_exclusive"));

  glewInfoFunc(fi, "glScissorExclusiveArrayvNV", glScissorExclusiveArrayvNV == NULL);
  glewInfoFunc(fi, "glScissorExclusiveNV", glScissorExclusiveNV == NULL);
}

#endif /* GL_NV_scissor_exclusive */

#ifdef GL_NV_shader_atomic_counters

static void _glewInfo_GL_NV_shader_atomic_counters (void)
{
  glewPrintExt("GL_NV_shader_atomic_counters", GLEW_NV_shader_atomic_counters, glewIsSupported("GL_NV_shader_atomic_counters"), glewGetExtension("GL_NV_shader_atomic_counters"));
}

#endif /* GL_NV_shader_atomic_counters */

#ifdef GL_NV_shader_atomic_float

static void _glewInfo_GL_NV_shader_atomic_float (void)
{
  glewPrintExt("GL_NV_shader_atomic_float", GLEW_NV_shader_atomic_float, glewIsSupported("GL_NV_shader_atomic_float"), glewGetExtension("GL_NV_shader_atomic_float"));
}

#endif /* GL_NV_shader_atomic_float */

#ifdef GL_NV_shader_atomic_float64

static void _glewInfo_GL_NV_shader_atomic_float64 (void)
{
  glewPrintExt("GL_NV_shader_atomic_float64", GLEW_NV_shader_atomic_float64, glewIsSupported("GL_NV_shader_atomic_float64"), glewGetExtension("GL_NV_shader_atomic_float64"));
}

#endif /* GL_NV_shader_atomic_float64 */

#ifdef GL_NV_shader_atomic_fp16_vector

static void _glewInfo_GL_NV_shader_atomic_fp16_vector (void)
{
  glewPrintExt("GL_NV_shader_atomic_fp16_vector", GLEW_NV_shader_atomic_fp16_vector, glewIsSupported("GL_NV_shader_atomic_fp16_vector"), glewGetExtension("GL_NV_shader_atomic_fp16_vector"));
}

#endif /* GL_NV_shader_atomic_fp16_vector */

#ifdef GL_NV_shader_atomic_int64

static void _glewInfo_GL_NV_shader_atomic_int64 (void)
{
  glewPrintExt("GL_NV_shader_atomic_int64", GLEW_NV_shader_atomic_int64, glewIsSupported("GL_NV_shader_atomic_int64"), glewGetExtension("GL_NV_shader_atomic_int64"));
}

#endif /* GL_NV_shader_atomic_int64 */

#ifdef GL_NV_shader_buffer_load

static void _glewInfo_GL_NV_shader_buffer_load (void)
{
  GLboolean fi = glewPrintExt("GL_NV_shader_buffer_load", GLEW_NV_shader_buffer_load, glewIsSupported("GL_NV_shader_buffer_load"), glewGetExtension("GL_NV_shader_buffer_load"));

  glewInfoFunc(fi, "glGetBufferParameterui64vNV", glGetBufferParameterui64vNV == NULL);
  glewInfoFunc(fi, "glGetIntegerui64vNV", glGetIntegerui64vNV == NULL);
  glewInfoFunc(fi, "glGetNamedBufferParameterui64vNV", glGetNamedBufferParameterui64vNV == NULL);
  glewInfoFunc(fi, "glIsBufferResidentNV", glIsBufferResidentNV == NULL);
  glewInfoFunc(fi, "glIsNamedBufferResidentNV", glIsNamedBufferResidentNV == NULL);
  glewInfoFunc(fi, "glMakeBufferNonResidentNV", glMakeBufferNonResidentNV == NULL);
  glewInfoFunc(fi, "glMakeBufferResidentNV", glMakeBufferResidentNV == NULL);
  glewInfoFunc(fi, "glMakeNamedBufferNonResidentNV", glMakeNamedBufferNonResidentNV == NULL);
  glewInfoFunc(fi, "glMakeNamedBufferResidentNV", glMakeNamedBufferResidentNV == NULL);
  glewInfoFunc(fi, "glProgramUniformui64NV", glProgramUniformui64NV == NULL);
  glewInfoFunc(fi, "glProgramUniformui64vNV", glProgramUniformui64vNV == NULL);
  glewInfoFunc(fi, "glUniformui64NV", glUniformui64NV == NULL);
  glewInfoFunc(fi, "glUniformui64vNV", glUniformui64vNV == NULL);
}

#endif /* GL_NV_shader_buffer_load */

#ifdef GL_NV_shader_noperspective_interpolation

static void _glewInfo_GL_NV_shader_noperspective_interpolation (void)
{
  glewPrintExt("GL_NV_shader_noperspective_interpolation", GLEW_NV_shader_noperspective_interpolation, glewIsSupported("GL_NV_shader_noperspective_interpolation"), glewGetExtension("GL_NV_shader_noperspective_interpolation"));
}

#endif /* GL_NV_shader_noperspective_interpolation */

#ifdef GL_NV_shader_storage_buffer_object

static void _glewInfo_GL_NV_shader_storage_buffer_object (void)
{
  glewPrintExt("GL_NV_shader_storage_buffer_object", GLEW_NV_shader_storage_buffer_object, glewIsSupported("GL_NV_shader_storage_buffer_object"), glewGetExtension("GL_NV_shader_storage_buffer_object"));
}

#endif /* GL_NV_shader_storage_buffer_object */

#ifdef GL_NV_shader_subgroup_partitioned

static void _glewInfo_GL_NV_shader_subgroup_partitioned (void)
{
  glewPrintExt("GL_NV_shader_subgroup_partitioned", GLEW_NV_shader_subgroup_partitioned, glewIsSupported("GL_NV_shader_subgroup_partitioned"), glewGetExtension("GL_NV_shader_subgroup_partitioned"));
}

#endif /* GL_NV_shader_subgroup_partitioned */

#ifdef GL_NV_shader_texture_footprint

static void _glewInfo_GL_NV_shader_texture_footprint (void)
{
  glewPrintExt("GL_NV_shader_texture_footprint", GLEW_NV_shader_texture_footprint, glewIsSupported("GL_NV_shader_texture_footprint"), glewGetExtension("GL_NV_shader_texture_footprint"));
}

#endif /* GL_NV_shader_texture_footprint */

#ifdef GL_NV_shader_thread_group

static void _glewInfo_GL_NV_shader_thread_group (void)
{
  glewPrintExt("GL_NV_shader_thread_group", GLEW_NV_shader_thread_group, glewIsSupported("GL_NV_shader_thread_group"), glewGetExtension("GL_NV_shader_thread_group"));
}

#endif /* GL_NV_shader_thread_group */

#ifdef GL_NV_shader_thread_shuffle

static void _glewInfo_GL_NV_shader_thread_shuffle (void)
{
  glewPrintExt("GL_NV_shader_thread_shuffle", GLEW_NV_shader_thread_shuffle, glewIsSupported("GL_NV_shader_thread_shuffle"), glewGetExtension("GL_NV_shader_thread_shuffle"));
}

#endif /* GL_NV_shader_thread_shuffle */

#ifdef GL_NV_shading_rate_image

static void _glewInfo_GL_NV_shading_rate_image (void)
{
  GLboolean fi = glewPrintExt("GL_NV_shading_rate_image", GLEW_NV_shading_rate_image, glewIsSupported("GL_NV_shading_rate_image"), glewGetExtension("GL_NV_shading_rate_image"));

  glewInfoFunc(fi, "glBindShadingRateImageNV", glBindShadingRateImageNV == NULL);
  glewInfoFunc(fi, "glGetShadingRateImagePaletteNV", glGetShadingRateImagePaletteNV == NULL);
  glewInfoFunc(fi, "glGetShadingRateSampleLocationivNV", glGetShadingRateSampleLocationivNV == NULL);
  glewInfoFunc(fi, "glShadingRateImageBarrierNV", glShadingRateImageBarrierNV == NULL);
  glewInfoFunc(fi, "glShadingRateImagePaletteNV", glShadingRateImagePaletteNV == NULL);
  glewInfoFunc(fi, "glShadingRateSampleOrderCustomNV", glShadingRateSampleOrderCustomNV == NULL);
}

#endif /* GL_NV_shading_rate_image */

#ifdef GL_NV_shadow_samplers_array

static void _glewInfo_GL_NV_shadow_samplers_array (void)
{
  glewPrintExt("GL_NV_shadow_samplers_array", GLEW_NV_shadow_samplers_array, glewIsSupported("GL_NV_shadow_samplers_array"), glewGetExtension("GL_NV_shadow_samplers_array"));
}

#endif /* GL_NV_shadow_samplers_array */

#ifdef GL_NV_shadow_samplers_cube

static void _glewInfo_GL_NV_shadow_samplers_cube (void)
{
  glewPrintExt("GL_NV_shadow_samplers_cube", GLEW_NV_shadow_samplers_cube, glewIsSupported("GL_NV_shadow_samplers_cube"), glewGetExtension("GL_NV_shadow_samplers_cube"));
}

#endif /* GL_NV_shadow_samplers_cube */

#ifdef GL_NV_stereo_view_rendering

static void _glewInfo_GL_NV_stereo_view_rendering (void)
{
  glewPrintExt("GL_NV_stereo_view_rendering", GLEW_NV_stereo_view_rendering, glewIsSupported("GL_NV_stereo_view_rendering"), glewGetExtension("GL_NV_stereo_view_rendering"));
}

#endif /* GL_NV_stereo_view_rendering */

#ifdef GL_NV_tessellation_program5

static void _glewInfo_GL_NV_tessellation_program5 (void)
{
  glewPrintExt("GL_NV_tessellation_program5", GLEW_NV_tessellation_program5, glewIsSupported("GL_NV_tessellation_program5"), glewGetExtension("GL_NV_gpu_program5"));
}

#endif /* GL_NV_tessellation_program5 */

#ifdef GL_NV_texgen_emboss

static void _glewInfo_GL_NV_texgen_emboss (void)
{
  glewPrintExt("GL_NV_texgen_emboss", GLEW_NV_texgen_emboss, glewIsSupported("GL_NV_texgen_emboss"), glewGetExtension("GL_NV_texgen_emboss"));
}

#endif /* GL_NV_texgen_emboss */

#ifdef GL_NV_texgen_reflection

static void _glewInfo_GL_NV_texgen_reflection (void)
{
  glewPrintExt("GL_NV_texgen_reflection", GLEW_NV_texgen_reflection, glewIsSupported("GL_NV_texgen_reflection"), glewGetExtension("GL_NV_texgen_reflection"));
}

#endif /* GL_NV_texgen_reflection */

#ifdef GL_NV_texture_array

static void _glewInfo_GL_NV_texture_array (void)
{
  GLboolean fi = glewPrintExt("GL_NV_texture_array", GLEW_NV_texture_array, glewIsSupported("GL_NV_texture_array"), glewGetExtension("GL_NV_texture_array"));

  glewInfoFunc(fi, "glCompressedTexImage3DNV", glCompressedTexImage3DNV == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage3DNV", glCompressedTexSubImage3DNV == NULL);
  glewInfoFunc(fi, "glCopyTexSubImage3DNV", glCopyTexSubImage3DNV == NULL);
  glewInfoFunc(fi, "glFramebufferTextureLayerNV", glFramebufferTextureLayerNV == NULL);
  glewInfoFunc(fi, "glTexImage3DNV", glTexImage3DNV == NULL);
  glewInfoFunc(fi, "glTexSubImage3DNV", glTexSubImage3DNV == NULL);
}

#endif /* GL_NV_texture_array */

#ifdef GL_NV_texture_barrier

static void _glewInfo_GL_NV_texture_barrier (void)
{
  GLboolean fi = glewPrintExt("GL_NV_texture_barrier", GLEW_NV_texture_barrier, glewIsSupported("GL_NV_texture_barrier"), glewGetExtension("GL_NV_texture_barrier"));

  glewInfoFunc(fi, "glTextureBarrierNV", glTextureBarrierNV == NULL);
}

#endif /* GL_NV_texture_barrier */

#ifdef GL_NV_texture_border_clamp

static void _glewInfo_GL_NV_texture_border_clamp (void)
{
  glewPrintExt("GL_NV_texture_border_clamp", GLEW_NV_texture_border_clamp, glewIsSupported("GL_NV_texture_border_clamp"), glewGetExtension("GL_NV_texture_border_clamp"));
}

#endif /* GL_NV_texture_border_clamp */

#ifdef GL_NV_texture_compression_latc

static void _glewInfo_GL_NV_texture_compression_latc (void)
{
  glewPrintExt("GL_NV_texture_compression_latc", GLEW_NV_texture_compression_latc, glewIsSupported("GL_NV_texture_compression_latc"), glewGetExtension("GL_NV_texture_compression_latc"));
}

#endif /* GL_NV_texture_compression_latc */

#ifdef GL_NV_texture_compression_s3tc

static void _glewInfo_GL_NV_texture_compression_s3tc (void)
{
  glewPrintExt("GL_NV_texture_compression_s3tc", GLEW_NV_texture_compression_s3tc, glewIsSupported("GL_NV_texture_compression_s3tc"), glewGetExtension("GL_NV_texture_compression_s3tc"));
}

#endif /* GL_NV_texture_compression_s3tc */

#ifdef GL_NV_texture_compression_s3tc_update

static void _glewInfo_GL_NV_texture_compression_s3tc_update (void)
{
  glewPrintExt("GL_NV_texture_compression_s3tc_update", GLEW_NV_texture_compression_s3tc_update, glewIsSupported("GL_NV_texture_compression_s3tc_update"), glewGetExtension("GL_NV_texture_compression_s3tc_update"));
}

#endif /* GL_NV_texture_compression_s3tc_update */

#ifdef GL_NV_texture_compression_vtc

static void _glewInfo_GL_NV_texture_compression_vtc (void)
{
  glewPrintExt("GL_NV_texture_compression_vtc", GLEW_NV_texture_compression_vtc, glewIsSupported("GL_NV_texture_compression_vtc"), glewGetExtension("GL_NV_texture_compression_vtc"));
}

#endif /* GL_NV_texture_compression_vtc */

#ifdef GL_NV_texture_env_combine4

static void _glewInfo_GL_NV_texture_env_combine4 (void)
{
  glewPrintExt("GL_NV_texture_env_combine4", GLEW_NV_texture_env_combine4, glewIsSupported("GL_NV_texture_env_combine4"), glewGetExtension("GL_NV_texture_env_combine4"));
}

#endif /* GL_NV_texture_env_combine4 */

#ifdef GL_NV_texture_expand_normal

static void _glewInfo_GL_NV_texture_expand_normal (void)
{
  glewPrintExt("GL_NV_texture_expand_normal", GLEW_NV_texture_expand_normal, glewIsSupported("GL_NV_texture_expand_normal"), glewGetExtension("GL_NV_texture_expand_normal"));
}

#endif /* GL_NV_texture_expand_normal */

#ifdef GL_NV_texture_multisample

static void _glewInfo_GL_NV_texture_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_NV_texture_multisample", GLEW_NV_texture_multisample, glewIsSupported("GL_NV_texture_multisample"), glewGetExtension("GL_NV_texture_multisample"));

  glewInfoFunc(fi, "glTexImage2DMultisampleCoverageNV", glTexImage2DMultisampleCoverageNV == NULL);
  glewInfoFunc(fi, "glTexImage3DMultisampleCoverageNV", glTexImage3DMultisampleCoverageNV == NULL);
  glewInfoFunc(fi, "glTextureImage2DMultisampleCoverageNV", glTextureImage2DMultisampleCoverageNV == NULL);
  glewInfoFunc(fi, "glTextureImage2DMultisampleNV", glTextureImage2DMultisampleNV == NULL);
  glewInfoFunc(fi, "glTextureImage3DMultisampleCoverageNV", glTextureImage3DMultisampleCoverageNV == NULL);
  glewInfoFunc(fi, "glTextureImage3DMultisampleNV", glTextureImage3DMultisampleNV == NULL);
}

#endif /* GL_NV_texture_multisample */

#ifdef GL_NV_texture_npot_2D_mipmap

static void _glewInfo_GL_NV_texture_npot_2D_mipmap (void)
{
  glewPrintExt("GL_NV_texture_npot_2D_mipmap", GLEW_NV_texture_npot_2D_mipmap, glewIsSupported("GL_NV_texture_npot_2D_mipmap"), glewGetExtension("GL_NV_texture_npot_2D_mipmap"));
}

#endif /* GL_NV_texture_npot_2D_mipmap */

#ifdef GL_NV_texture_rectangle

static void _glewInfo_GL_NV_texture_rectangle (void)
{
  glewPrintExt("GL_NV_texture_rectangle", GLEW_NV_texture_rectangle, glewIsSupported("GL_NV_texture_rectangle"), glewGetExtension("GL_NV_texture_rectangle"));
}

#endif /* GL_NV_texture_rectangle */

#ifdef GL_NV_texture_rectangle_compressed

static void _glewInfo_GL_NV_texture_rectangle_compressed (void)
{
  glewPrintExt("GL_NV_texture_rectangle_compressed", GLEW_NV_texture_rectangle_compressed, glewIsSupported("GL_NV_texture_rectangle_compressed"), glewGetExtension("GL_NV_texture_rectangle_compressed"));
}

#endif /* GL_NV_texture_rectangle_compressed */

#ifdef GL_NV_texture_shader

static void _glewInfo_GL_NV_texture_shader (void)
{
  glewPrintExt("GL_NV_texture_shader", GLEW_NV_texture_shader, glewIsSupported("GL_NV_texture_shader"), glewGetExtension("GL_NV_texture_shader"));
}

#endif /* GL_NV_texture_shader */

#ifdef GL_NV_texture_shader2

static void _glewInfo_GL_NV_texture_shader2 (void)
{
  glewPrintExt("GL_NV_texture_shader2", GLEW_NV_texture_shader2, glewIsSupported("GL_NV_texture_shader2"), glewGetExtension("GL_NV_texture_shader2"));
}

#endif /* GL_NV_texture_shader2 */

#ifdef GL_NV_texture_shader3

static void _glewInfo_GL_NV_texture_shader3 (void)
{
  glewPrintExt("GL_NV_texture_shader3", GLEW_NV_texture_shader3, glewIsSupported("GL_NV_texture_shader3"), glewGetExtension("GL_NV_texture_shader3"));
}

#endif /* GL_NV_texture_shader3 */

#ifdef GL_NV_transform_feedback

static void _glewInfo_GL_NV_transform_feedback (void)
{
  GLboolean fi = glewPrintExt("GL_NV_transform_feedback", GLEW_NV_transform_feedback, glewIsSupported("GL_NV_transform_feedback"), glewGetExtension("GL_NV_transform_feedback"));

  glewInfoFunc(fi, "glActiveVaryingNV", glActiveVaryingNV == NULL);
  glewInfoFunc(fi, "glBeginTransformFeedbackNV", glBeginTransformFeedbackNV == NULL);
  glewInfoFunc(fi, "glBindBufferBaseNV", glBindBufferBaseNV == NULL);
  glewInfoFunc(fi, "glBindBufferOffsetNV", glBindBufferOffsetNV == NULL);
  glewInfoFunc(fi, "glBindBufferRangeNV", glBindBufferRangeNV == NULL);
  glewInfoFunc(fi, "glEndTransformFeedbackNV", glEndTransformFeedbackNV == NULL);
  glewInfoFunc(fi, "glGetActiveVaryingNV", glGetActiveVaryingNV == NULL);
  glewInfoFunc(fi, "glGetTransformFeedbackVaryingNV", glGetTransformFeedbackVaryingNV == NULL);
  glewInfoFunc(fi, "glGetVaryingLocationNV", glGetVaryingLocationNV == NULL);
  glewInfoFunc(fi, "glTransformFeedbackAttribsNV", glTransformFeedbackAttribsNV == NULL);
  glewInfoFunc(fi, "glTransformFeedbackVaryingsNV", glTransformFeedbackVaryingsNV == NULL);
}

#endif /* GL_NV_transform_feedback */

#ifdef GL_NV_transform_feedback2

static void _glewInfo_GL_NV_transform_feedback2 (void)
{
  GLboolean fi = glewPrintExt("GL_NV_transform_feedback2", GLEW_NV_transform_feedback2, glewIsSupported("GL_NV_transform_feedback2"), glewGetExtension("GL_NV_transform_feedback2"));

  glewInfoFunc(fi, "glBindTransformFeedbackNV", glBindTransformFeedbackNV == NULL);
  glewInfoFunc(fi, "glDeleteTransformFeedbacksNV", glDeleteTransformFeedbacksNV == NULL);
  glewInfoFunc(fi, "glDrawTransformFeedbackNV", glDrawTransformFeedbackNV == NULL);
  glewInfoFunc(fi, "glGenTransformFeedbacksNV", glGenTransformFeedbacksNV == NULL);
  glewInfoFunc(fi, "glIsTransformFeedbackNV", glIsTransformFeedbackNV == NULL);
  glewInfoFunc(fi, "glPauseTransformFeedbackNV", glPauseTransformFeedbackNV == NULL);
  glewInfoFunc(fi, "glResumeTransformFeedbackNV", glResumeTransformFeedbackNV == NULL);
}

#endif /* GL_NV_transform_feedback2 */

#ifdef GL_NV_uniform_buffer_unified_memory

static void _glewInfo_GL_NV_uniform_buffer_unified_memory (void)
{
  glewPrintExt("GL_NV_uniform_buffer_unified_memory", GLEW_NV_uniform_buffer_unified_memory, glewIsSupported("GL_NV_uniform_buffer_unified_memory"), glewGetExtension("GL_NV_uniform_buffer_unified_memory"));
}

#endif /* GL_NV_uniform_buffer_unified_memory */

#ifdef GL_NV_vdpau_interop

static void _glewInfo_GL_NV_vdpau_interop (void)
{
  GLboolean fi = glewPrintExt("GL_NV_vdpau_interop", GLEW_NV_vdpau_interop, glewIsSupported("GL_NV_vdpau_interop"), glewGetExtension("GL_NV_vdpau_interop"));

  glewInfoFunc(fi, "glVDPAUFiniNV", glVDPAUFiniNV == NULL);
  glewInfoFunc(fi, "glVDPAUGetSurfaceivNV", glVDPAUGetSurfaceivNV == NULL);
  glewInfoFunc(fi, "glVDPAUInitNV", glVDPAUInitNV == NULL);
  glewInfoFunc(fi, "glVDPAUIsSurfaceNV", glVDPAUIsSurfaceNV == NULL);
  glewInfoFunc(fi, "glVDPAUMapSurfacesNV", glVDPAUMapSurfacesNV == NULL);
  glewInfoFunc(fi, "glVDPAURegisterOutputSurfaceNV", glVDPAURegisterOutputSurfaceNV == NULL);
  glewInfoFunc(fi, "glVDPAURegisterVideoSurfaceNV", glVDPAURegisterVideoSurfaceNV == NULL);
  glewInfoFunc(fi, "glVDPAUSurfaceAccessNV", glVDPAUSurfaceAccessNV == NULL);
  glewInfoFunc(fi, "glVDPAUUnmapSurfacesNV", glVDPAUUnmapSurfacesNV == NULL);
  glewInfoFunc(fi, "glVDPAUUnregisterSurfaceNV", glVDPAUUnregisterSurfaceNV == NULL);
}

#endif /* GL_NV_vdpau_interop */

#ifdef GL_NV_vdpau_interop2

static void _glewInfo_GL_NV_vdpau_interop2 (void)
{
  GLboolean fi = glewPrintExt("GL_NV_vdpau_interop2", GLEW_NV_vdpau_interop2, glewIsSupported("GL_NV_vdpau_interop2"), glewGetExtension("GL_NV_vdpau_interop2"));

  glewInfoFunc(fi, "glVDPAURegisterVideoSurfaceWithPictureStructureNV", glVDPAURegisterVideoSurfaceWithPictureStructureNV == NULL);
}

#endif /* GL_NV_vdpau_interop2 */

#ifdef GL_NV_vertex_array_range

static void _glewInfo_GL_NV_vertex_array_range (void)
{
  GLboolean fi = glewPrintExt("GL_NV_vertex_array_range", GLEW_NV_vertex_array_range, glewIsSupported("GL_NV_vertex_array_range"), glewGetExtension("GL_NV_vertex_array_range"));

  glewInfoFunc(fi, "glFlushVertexArrayRangeNV", glFlushVertexArrayRangeNV == NULL);
  glewInfoFunc(fi, "glVertexArrayRangeNV", glVertexArrayRangeNV == NULL);
}

#endif /* GL_NV_vertex_array_range */

#ifdef GL_NV_vertex_array_range2

static void _glewInfo_GL_NV_vertex_array_range2 (void)
{
  glewPrintExt("GL_NV_vertex_array_range2", GLEW_NV_vertex_array_range2, glewIsSupported("GL_NV_vertex_array_range2"), glewGetExtension("GL_NV_vertex_array_range2"));
}

#endif /* GL_NV_vertex_array_range2 */

#ifdef GL_NV_vertex_attrib_integer_64bit

static void _glewInfo_GL_NV_vertex_attrib_integer_64bit (void)
{
  GLboolean fi = glewPrintExt("GL_NV_vertex_attrib_integer_64bit", GLEW_NV_vertex_attrib_integer_64bit, glewIsSupported("GL_NV_vertex_attrib_integer_64bit"), glewGetExtension("GL_NV_vertex_attrib_integer_64bit"));

  glewInfoFunc(fi, "glGetVertexAttribLi64vNV", glGetVertexAttribLi64vNV == NULL);
  glewInfoFunc(fi, "glGetVertexAttribLui64vNV", glGetVertexAttribLui64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL1i64NV", glVertexAttribL1i64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL1i64vNV", glVertexAttribL1i64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL1ui64NV", glVertexAttribL1ui64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL1ui64vNV", glVertexAttribL1ui64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL2i64NV", glVertexAttribL2i64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL2i64vNV", glVertexAttribL2i64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL2ui64NV", glVertexAttribL2ui64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL2ui64vNV", glVertexAttribL2ui64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL3i64NV", glVertexAttribL3i64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL3i64vNV", glVertexAttribL3i64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL3ui64NV", glVertexAttribL3ui64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL3ui64vNV", glVertexAttribL3ui64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL4i64NV", glVertexAttribL4i64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL4i64vNV", glVertexAttribL4i64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribL4ui64NV", glVertexAttribL4ui64NV == NULL);
  glewInfoFunc(fi, "glVertexAttribL4ui64vNV", glVertexAttribL4ui64vNV == NULL);
  glewInfoFunc(fi, "glVertexAttribLFormatNV", glVertexAttribLFormatNV == NULL);
}

#endif /* GL_NV_vertex_attrib_integer_64bit */

#ifdef GL_NV_vertex_buffer_unified_memory

static void _glewInfo_GL_NV_vertex_buffer_unified_memory (void)
{
  GLboolean fi = glewPrintExt("GL_NV_vertex_buffer_unified_memory", GLEW_NV_vertex_buffer_unified_memory, glewIsSupported("GL_NV_vertex_buffer_unified_memory"), glewGetExtension("GL_NV_vertex_buffer_unified_memory"));

  glewInfoFunc(fi, "glBufferAddressRangeNV", glBufferAddressRangeNV == NULL);
  glewInfoFunc(fi, "glColorFormatNV", glColorFormatNV == NULL);
  glewInfoFunc(fi, "glEdgeFlagFormatNV", glEdgeFlagFormatNV == NULL);
  glewInfoFunc(fi, "glFogCoordFormatNV", glFogCoordFormatNV == NULL);
  glewInfoFunc(fi, "glGetIntegerui64i_vNV", glGetIntegerui64i_vNV == NULL);
  glewInfoFunc(fi, "glIndexFormatNV", glIndexFormatNV == NULL);
  glewInfoFunc(fi, "glNormalFormatNV", glNormalFormatNV == NULL);
  glewInfoFunc(fi, "glSecondaryColorFormatNV", glSecondaryColorFormatNV == NULL);
  glewInfoFunc(fi, "glTexCoordFormatNV", glTexCoordFormatNV == NULL);
  glewInfoFunc(fi, "glVertexAttribFormatNV", glVertexAttribFormatNV == NULL);
  glewInfoFunc(fi, "glVertexAttribIFormatNV", glVertexAttribIFormatNV == NULL);
  glewInfoFunc(fi, "glVertexFormatNV", glVertexFormatNV == NULL);
}

#endif /* GL_NV_vertex_buffer_unified_memory */

#ifdef GL_NV_vertex_program

static void _glewInfo_GL_NV_vertex_program (void)
{
  GLboolean fi = glewPrintExt("GL_NV_vertex_program", GLEW_NV_vertex_program, glewIsSupported("GL_NV_vertex_program"), glewGetExtension("GL_NV_vertex_program"));

  glewInfoFunc(fi, "glAreProgramsResidentNV", glAreProgramsResidentNV == NULL);
  glewInfoFunc(fi, "glBindProgramNV", glBindProgramNV == NULL);
  glewInfoFunc(fi, "glDeleteProgramsNV", glDeleteProgramsNV == NULL);
  glewInfoFunc(fi, "glExecuteProgramNV", glExecuteProgramNV == NULL);
  glewInfoFunc(fi, "glGenProgramsNV", glGenProgramsNV == NULL);
  glewInfoFunc(fi, "glGetProgramParameterdvNV", glGetProgramParameterdvNV == NULL);
  glewInfoFunc(fi, "glGetProgramParameterfvNV", glGetProgramParameterfvNV == NULL);
  glewInfoFunc(fi, "glGetProgramStringNV", glGetProgramStringNV == NULL);
  glewInfoFunc(fi, "glGetProgramivNV", glGetProgramivNV == NULL);
  glewInfoFunc(fi, "glGetTrackMatrixivNV", glGetTrackMatrixivNV == NULL);
  glewInfoFunc(fi, "glGetVertexAttribPointervNV", glGetVertexAttribPointervNV == NULL);
  glewInfoFunc(fi, "glGetVertexAttribdvNV", glGetVertexAttribdvNV == NULL);
  glewInfoFunc(fi, "glGetVertexAttribfvNV", glGetVertexAttribfvNV == NULL);
  glewInfoFunc(fi, "glGetVertexAttribivNV", glGetVertexAttribivNV == NULL);
  glewInfoFunc(fi, "glIsProgramNV", glIsProgramNV == NULL);
  glewInfoFunc(fi, "glLoadProgramNV", glLoadProgramNV == NULL);
  glewInfoFunc(fi, "glProgramParameter4dNV", glProgramParameter4dNV == NULL);
  glewInfoFunc(fi, "glProgramParameter4dvNV", glProgramParameter4dvNV == NULL);
  glewInfoFunc(fi, "glProgramParameter4fNV", glProgramParameter4fNV == NULL);
  glewInfoFunc(fi, "glProgramParameter4fvNV", glProgramParameter4fvNV == NULL);
  glewInfoFunc(fi, "glProgramParameters4dvNV", glProgramParameters4dvNV == NULL);
  glewInfoFunc(fi, "glProgramParameters4fvNV", glProgramParameters4fvNV == NULL);
  glewInfoFunc(fi, "glRequestResidentProgramsNV", glRequestResidentProgramsNV == NULL);
  glewInfoFunc(fi, "glTrackMatrixNV", glTrackMatrixNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1dNV", glVertexAttrib1dNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1dvNV", glVertexAttrib1dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1fNV", glVertexAttrib1fNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1fvNV", glVertexAttrib1fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1sNV", glVertexAttrib1sNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib1svNV", glVertexAttrib1svNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2dNV", glVertexAttrib2dNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2dvNV", glVertexAttrib2dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2fNV", glVertexAttrib2fNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2fvNV", glVertexAttrib2fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2sNV", glVertexAttrib2sNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib2svNV", glVertexAttrib2svNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3dNV", glVertexAttrib3dNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3dvNV", glVertexAttrib3dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3fNV", glVertexAttrib3fNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3fvNV", glVertexAttrib3fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3sNV", glVertexAttrib3sNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib3svNV", glVertexAttrib3svNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4dNV", glVertexAttrib4dNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4dvNV", glVertexAttrib4dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4fNV", glVertexAttrib4fNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4fvNV", glVertexAttrib4fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4sNV", glVertexAttrib4sNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4svNV", glVertexAttrib4svNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4ubNV", glVertexAttrib4ubNV == NULL);
  glewInfoFunc(fi, "glVertexAttrib4ubvNV", glVertexAttrib4ubvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribPointerNV", glVertexAttribPointerNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs1dvNV", glVertexAttribs1dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs1fvNV", glVertexAttribs1fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs1svNV", glVertexAttribs1svNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs2dvNV", glVertexAttribs2dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs2fvNV", glVertexAttribs2fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs2svNV", glVertexAttribs2svNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs3dvNV", glVertexAttribs3dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs3fvNV", glVertexAttribs3fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs3svNV", glVertexAttribs3svNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs4dvNV", glVertexAttribs4dvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs4fvNV", glVertexAttribs4fvNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs4svNV", glVertexAttribs4svNV == NULL);
  glewInfoFunc(fi, "glVertexAttribs4ubvNV", glVertexAttribs4ubvNV == NULL);
}

#endif /* GL_NV_vertex_program */

#ifdef GL_NV_vertex_program1_1

static void _glewInfo_GL_NV_vertex_program1_1 (void)
{
  glewPrintExt("GL_NV_vertex_program1_1", GLEW_NV_vertex_program1_1, glewIsSupported("GL_NV_vertex_program1_1"), glewGetExtension("GL_NV_vertex_program1_1"));
}

#endif /* GL_NV_vertex_program1_1 */

#ifdef GL_NV_vertex_program2

static void _glewInfo_GL_NV_vertex_program2 (void)
{
  glewPrintExt("GL_NV_vertex_program2", GLEW_NV_vertex_program2, glewIsSupported("GL_NV_vertex_program2"), glewGetExtension("GL_NV_vertex_program2"));
}

#endif /* GL_NV_vertex_program2 */

#ifdef GL_NV_vertex_program2_option

static void _glewInfo_GL_NV_vertex_program2_option (void)
{
  glewPrintExt("GL_NV_vertex_program2_option", GLEW_NV_vertex_program2_option, glewIsSupported("GL_NV_vertex_program2_option"), glewGetExtension("GL_NV_vertex_program2_option"));
}

#endif /* GL_NV_vertex_program2_option */

#ifdef GL_NV_vertex_program3

static void _glewInfo_GL_NV_vertex_program3 (void)
{
  glewPrintExt("GL_NV_vertex_program3", GLEW_NV_vertex_program3, glewIsSupported("GL_NV_vertex_program3"), glewGetExtension("GL_NV_vertex_program3"));
}

#endif /* GL_NV_vertex_program3 */

#ifdef GL_NV_vertex_program4

static void _glewInfo_GL_NV_vertex_program4 (void)
{
  glewPrintExt("GL_NV_vertex_program4", GLEW_NV_vertex_program4, glewIsSupported("GL_NV_vertex_program4"), glewGetExtension("GL_NV_gpu_program4"));
}

#endif /* GL_NV_vertex_program4 */

#ifdef GL_NV_video_capture

static void _glewInfo_GL_NV_video_capture (void)
{
  GLboolean fi = glewPrintExt("GL_NV_video_capture", GLEW_NV_video_capture, glewIsSupported("GL_NV_video_capture"), glewGetExtension("GL_NV_video_capture"));

  glewInfoFunc(fi, "glBeginVideoCaptureNV", glBeginVideoCaptureNV == NULL);
  glewInfoFunc(fi, "glBindVideoCaptureStreamBufferNV", glBindVideoCaptureStreamBufferNV == NULL);
  glewInfoFunc(fi, "glBindVideoCaptureStreamTextureNV", glBindVideoCaptureStreamTextureNV == NULL);
  glewInfoFunc(fi, "glEndVideoCaptureNV", glEndVideoCaptureNV == NULL);
  glewInfoFunc(fi, "glGetVideoCaptureStreamdvNV", glGetVideoCaptureStreamdvNV == NULL);
  glewInfoFunc(fi, "glGetVideoCaptureStreamfvNV", glGetVideoCaptureStreamfvNV == NULL);
  glewInfoFunc(fi, "glGetVideoCaptureStreamivNV", glGetVideoCaptureStreamivNV == NULL);
  glewInfoFunc(fi, "glGetVideoCaptureivNV", glGetVideoCaptureivNV == NULL);
  glewInfoFunc(fi, "glVideoCaptureNV", glVideoCaptureNV == NULL);
  glewInfoFunc(fi, "glVideoCaptureStreamParameterdvNV", glVideoCaptureStreamParameterdvNV == NULL);
  glewInfoFunc(fi, "glVideoCaptureStreamParameterfvNV", glVideoCaptureStreamParameterfvNV == NULL);
  glewInfoFunc(fi, "glVideoCaptureStreamParameterivNV", glVideoCaptureStreamParameterivNV == NULL);
}

#endif /* GL_NV_video_capture */

#ifdef GL_NV_viewport_array

static void _glewInfo_GL_NV_viewport_array (void)
{
  GLboolean fi = glewPrintExt("GL_NV_viewport_array", GLEW_NV_viewport_array, glewIsSupported("GL_NV_viewport_array"), glewGetExtension("GL_NV_viewport_array"));

  glewInfoFunc(fi, "glDepthRangeArrayfvNV", glDepthRangeArrayfvNV == NULL);
  glewInfoFunc(fi, "glDepthRangeIndexedfNV", glDepthRangeIndexedfNV == NULL);
  glewInfoFunc(fi, "glDisableiNV", glDisableiNV == NULL);
  glewInfoFunc(fi, "glEnableiNV", glEnableiNV == NULL);
  glewInfoFunc(fi, "glGetFloati_vNV", glGetFloati_vNV == NULL);
  glewInfoFunc(fi, "glIsEnablediNV", glIsEnablediNV == NULL);
  glewInfoFunc(fi, "glScissorArrayvNV", glScissorArrayvNV == NULL);
  glewInfoFunc(fi, "glScissorIndexedNV", glScissorIndexedNV == NULL);
  glewInfoFunc(fi, "glScissorIndexedvNV", glScissorIndexedvNV == NULL);
  glewInfoFunc(fi, "glViewportArrayvNV", glViewportArrayvNV == NULL);
  glewInfoFunc(fi, "glViewportIndexedfNV", glViewportIndexedfNV == NULL);
  glewInfoFunc(fi, "glViewportIndexedfvNV", glViewportIndexedfvNV == NULL);
}

#endif /* GL_NV_viewport_array */

#ifdef GL_NV_viewport_array2

static void _glewInfo_GL_NV_viewport_array2 (void)
{
  glewPrintExt("GL_NV_viewport_array2", GLEW_NV_viewport_array2, glewIsSupported("GL_NV_viewport_array2"), glewGetExtension("GL_NV_viewport_array2"));
}

#endif /* GL_NV_viewport_array2 */

#ifdef GL_NV_viewport_swizzle

static void _glewInfo_GL_NV_viewport_swizzle (void)
{
  GLboolean fi = glewPrintExt("GL_NV_viewport_swizzle", GLEW_NV_viewport_swizzle, glewIsSupported("GL_NV_viewport_swizzle"), glewGetExtension("GL_NV_viewport_swizzle"));

  glewInfoFunc(fi, "glViewportSwizzleNV", glViewportSwizzleNV == NULL);
}

#endif /* GL_NV_viewport_swizzle */

#ifdef GL_OES_EGL_image

static void _glewInfo_GL_OES_EGL_image (void)
{
  GLboolean fi = glewPrintExt("GL_OES_EGL_image", GLEW_OES_EGL_image, glewIsSupported("GL_OES_EGL_image"), glewGetExtension("GL_OES_EGL_image"));

  glewInfoFunc(fi, "glEGLImageTargetRenderbufferStorageOES", glEGLImageTargetRenderbufferStorageOES == NULL);
  glewInfoFunc(fi, "glEGLImageTargetTexture2DOES", glEGLImageTargetTexture2DOES == NULL);
}

#endif /* GL_OES_EGL_image */

#ifdef GL_OES_EGL_image_external

static void _glewInfo_GL_OES_EGL_image_external (void)
{
  glewPrintExt("GL_OES_EGL_image_external", GLEW_OES_EGL_image_external, glewIsSupported("GL_OES_EGL_image_external"), glewGetExtension("GL_OES_EGL_image_external"));
}

#endif /* GL_OES_EGL_image_external */

#ifdef GL_OES_EGL_image_external_essl3

static void _glewInfo_GL_OES_EGL_image_external_essl3 (void)
{
  glewPrintExt("GL_OES_EGL_image_external_essl3", GLEW_OES_EGL_image_external_essl3, glewIsSupported("GL_OES_EGL_image_external_essl3"), glewGetExtension("GL_OES_EGL_image_external_essl3"));
}

#endif /* GL_OES_EGL_image_external_essl3 */

#ifdef GL_OES_blend_equation_separate

static void _glewInfo_GL_OES_blend_equation_separate (void)
{
  GLboolean fi = glewPrintExt("GL_OES_blend_equation_separate", GLEW_OES_blend_equation_separate, glewIsSupported("GL_OES_blend_equation_separate"), glewGetExtension("GL_OES_blend_equation_separate"));

  glewInfoFunc(fi, "glBlendEquationSeparateOES", glBlendEquationSeparateOES == NULL);
}

#endif /* GL_OES_blend_equation_separate */

#ifdef GL_OES_blend_func_separate

static void _glewInfo_GL_OES_blend_func_separate (void)
{
  GLboolean fi = glewPrintExt("GL_OES_blend_func_separate", GLEW_OES_blend_func_separate, glewIsSupported("GL_OES_blend_func_separate"), glewGetExtension("GL_OES_blend_func_separate"));

  glewInfoFunc(fi, "glBlendFuncSeparateOES", glBlendFuncSeparateOES == NULL);
}

#endif /* GL_OES_blend_func_separate */

#ifdef GL_OES_blend_subtract

static void _glewInfo_GL_OES_blend_subtract (void)
{
  GLboolean fi = glewPrintExt("GL_OES_blend_subtract", GLEW_OES_blend_subtract, glewIsSupported("GL_OES_blend_subtract"), glewGetExtension("GL_OES_blend_subtract"));

  glewInfoFunc(fi, "glBlendEquationOES", glBlendEquationOES == NULL);
}

#endif /* GL_OES_blend_subtract */

#ifdef GL_OES_byte_coordinates

static void _glewInfo_GL_OES_byte_coordinates (void)
{
  glewPrintExt("GL_OES_byte_coordinates", GLEW_OES_byte_coordinates, glewIsSupported("GL_OES_byte_coordinates"), glewGetExtension("GL_OES_byte_coordinates"));
}

#endif /* GL_OES_byte_coordinates */

#ifdef GL_OES_compressed_ETC1_RGB8_texture

static void _glewInfo_GL_OES_compressed_ETC1_RGB8_texture (void)
{
  glewPrintExt("GL_OES_compressed_ETC1_RGB8_texture", GLEW_OES_compressed_ETC1_RGB8_texture, glewIsSupported("GL_OES_compressed_ETC1_RGB8_texture"), glewGetExtension("GL_OES_compressed_ETC1_RGB8_texture"));
}

#endif /* GL_OES_compressed_ETC1_RGB8_texture */

#ifdef GL_OES_compressed_paletted_texture

static void _glewInfo_GL_OES_compressed_paletted_texture (void)
{
  glewPrintExt("GL_OES_compressed_paletted_texture", GLEW_OES_compressed_paletted_texture, glewIsSupported("GL_OES_compressed_paletted_texture"), glewGetExtension("GL_OES_compressed_paletted_texture"));
}

#endif /* GL_OES_compressed_paletted_texture */

#ifdef GL_OES_copy_image

static void _glewInfo_GL_OES_copy_image (void)
{
  GLboolean fi = glewPrintExt("GL_OES_copy_image", GLEW_OES_copy_image, glewIsSupported("GL_OES_copy_image"), glewGetExtension("GL_OES_copy_image"));

  glewInfoFunc(fi, "glCopyImageSubDataOES", glCopyImageSubDataOES == NULL);
}

#endif /* GL_OES_copy_image */

#ifdef GL_OES_depth24

static void _glewInfo_GL_OES_depth24 (void)
{
  glewPrintExt("GL_OES_depth24", GLEW_OES_depth24, glewIsSupported("GL_OES_depth24"), glewGetExtension("GL_OES_depth24"));
}

#endif /* GL_OES_depth24 */

#ifdef GL_OES_depth32

static void _glewInfo_GL_OES_depth32 (void)
{
  glewPrintExt("GL_OES_depth32", GLEW_OES_depth32, glewIsSupported("GL_OES_depth32"), glewGetExtension("GL_OES_depth32"));
}

#endif /* GL_OES_depth32 */

#ifdef GL_OES_depth_texture

static void _glewInfo_GL_OES_depth_texture (void)
{
  glewPrintExt("GL_OES_depth_texture", GLEW_OES_depth_texture, glewIsSupported("GL_OES_depth_texture"), glewGetExtension("GL_OES_depth_texture"));
}

#endif /* GL_OES_depth_texture */

#ifdef GL_OES_depth_texture_cube_map

static void _glewInfo_GL_OES_depth_texture_cube_map (void)
{
  glewPrintExt("GL_OES_depth_texture_cube_map", GLEW_OES_depth_texture_cube_map, glewIsSupported("GL_OES_depth_texture_cube_map"), glewGetExtension("GL_OES_depth_texture_cube_map"));
}

#endif /* GL_OES_depth_texture_cube_map */

#ifdef GL_OES_draw_buffers_indexed

static void _glewInfo_GL_OES_draw_buffers_indexed (void)
{
  GLboolean fi = glewPrintExt("GL_OES_draw_buffers_indexed", GLEW_OES_draw_buffers_indexed, glewIsSupported("GL_OES_draw_buffers_indexed"), glewGetExtension("GL_OES_draw_buffers_indexed"));

  glewInfoFunc(fi, "glBlendEquationSeparateiOES", glBlendEquationSeparateiOES == NULL);
  glewInfoFunc(fi, "glBlendEquationiOES", glBlendEquationiOES == NULL);
  glewInfoFunc(fi, "glBlendFuncSeparateiOES", glBlendFuncSeparateiOES == NULL);
  glewInfoFunc(fi, "glBlendFunciOES", glBlendFunciOES == NULL);
  glewInfoFunc(fi, "glColorMaskiOES", glColorMaskiOES == NULL);
  glewInfoFunc(fi, "glDisableiOES", glDisableiOES == NULL);
  glewInfoFunc(fi, "glEnableiOES", glEnableiOES == NULL);
  glewInfoFunc(fi, "glIsEnablediOES", glIsEnablediOES == NULL);
}

#endif /* GL_OES_draw_buffers_indexed */

#ifdef GL_OES_draw_texture

static void _glewInfo_GL_OES_draw_texture (void)
{
  glewPrintExt("GL_OES_draw_texture", GLEW_OES_draw_texture, glewIsSupported("GL_OES_draw_texture"), glewGetExtension("GL_OES_draw_texture"));
}

#endif /* GL_OES_draw_texture */

#ifdef GL_OES_element_index_uint

static void _glewInfo_GL_OES_element_index_uint (void)
{
  glewPrintExt("GL_OES_element_index_uint", GLEW_OES_element_index_uint, glewIsSupported("GL_OES_element_index_uint"), glewGetExtension("GL_OES_element_index_uint"));
}

#endif /* GL_OES_element_index_uint */

#ifdef GL_OES_extended_matrix_palette

static void _glewInfo_GL_OES_extended_matrix_palette (void)
{
  glewPrintExt("GL_OES_extended_matrix_palette", GLEW_OES_extended_matrix_palette, glewIsSupported("GL_OES_extended_matrix_palette"), glewGetExtension("GL_OES_extended_matrix_palette"));
}

#endif /* GL_OES_extended_matrix_palette */

#ifdef GL_OES_fbo_render_mipmap

static void _glewInfo_GL_OES_fbo_render_mipmap (void)
{
  glewPrintExt("GL_OES_fbo_render_mipmap", GLEW_OES_fbo_render_mipmap, glewIsSupported("GL_OES_fbo_render_mipmap"), glewGetExtension("GL_OES_fbo_render_mipmap"));
}

#endif /* GL_OES_fbo_render_mipmap */

#ifdef GL_OES_fragment_precision_high

static void _glewInfo_GL_OES_fragment_precision_high (void)
{
  glewPrintExt("GL_OES_fragment_precision_high", GLEW_OES_fragment_precision_high, glewIsSupported("GL_OES_fragment_precision_high"), glewGetExtension("GL_OES_fragment_precision_high"));
}

#endif /* GL_OES_fragment_precision_high */

#ifdef GL_OES_framebuffer_object

static void _glewInfo_GL_OES_framebuffer_object (void)
{
  GLboolean fi = glewPrintExt("GL_OES_framebuffer_object", GLEW_OES_framebuffer_object, glewIsSupported("GL_OES_framebuffer_object"), glewGetExtension("GL_OES_framebuffer_object"));

  glewInfoFunc(fi, "glBindFramebufferOES", glBindFramebufferOES == NULL);
  glewInfoFunc(fi, "glBindRenderbufferOES", glBindRenderbufferOES == NULL);
  glewInfoFunc(fi, "glCheckFramebufferStatusOES", glCheckFramebufferStatusOES == NULL);
  glewInfoFunc(fi, "glDeleteFramebuffersOES", glDeleteFramebuffersOES == NULL);
  glewInfoFunc(fi, "glDeleteRenderbuffersOES", glDeleteRenderbuffersOES == NULL);
  glewInfoFunc(fi, "glFramebufferRenderbufferOES", glFramebufferRenderbufferOES == NULL);
  glewInfoFunc(fi, "glFramebufferTexture2DOES", glFramebufferTexture2DOES == NULL);
  glewInfoFunc(fi, "glGenFramebuffersOES", glGenFramebuffersOES == NULL);
  glewInfoFunc(fi, "glGenRenderbuffersOES", glGenRenderbuffersOES == NULL);
  glewInfoFunc(fi, "glGenerateMipmapOES", glGenerateMipmapOES == NULL);
  glewInfoFunc(fi, "glGetFramebufferAttachmentParameterivOES", glGetFramebufferAttachmentParameterivOES == NULL);
  glewInfoFunc(fi, "glGetRenderbufferParameterivOES", glGetRenderbufferParameterivOES == NULL);
  glewInfoFunc(fi, "glIsFramebufferOES", glIsFramebufferOES == NULL);
  glewInfoFunc(fi, "glIsRenderbufferOES", glIsRenderbufferOES == NULL);
  glewInfoFunc(fi, "glRenderbufferStorageOES", glRenderbufferStorageOES == NULL);
}

#endif /* GL_OES_framebuffer_object */

#ifdef GL_OES_geometry_point_size

static void _glewInfo_GL_OES_geometry_point_size (void)
{
  glewPrintExt("GL_OES_geometry_point_size", GLEW_OES_geometry_point_size, glewIsSupported("GL_OES_geometry_point_size"), glewGetExtension("GL_OES_geometry_point_size"));
}

#endif /* GL_OES_geometry_point_size */

#ifdef GL_OES_geometry_shader

static void _glewInfo_GL_OES_geometry_shader (void)
{
  glewPrintExt("GL_OES_geometry_shader", GLEW_OES_geometry_shader, glewIsSupported("GL_OES_geometry_shader"), glewGetExtension("GL_OES_geometry_shader"));
}

#endif /* GL_OES_geometry_shader */

#ifdef GL_OES_get_program_binary

static void _glewInfo_GL_OES_get_program_binary (void)
{
  GLboolean fi = glewPrintExt("GL_OES_get_program_binary", GLEW_OES_get_program_binary, glewIsSupported("GL_OES_get_program_binary"), glewGetExtension("GL_OES_get_program_binary"));

  glewInfoFunc(fi, "glGetProgramBinaryOES", glGetProgramBinaryOES == NULL);
  glewInfoFunc(fi, "glProgramBinaryOES", glProgramBinaryOES == NULL);
}

#endif /* GL_OES_get_program_binary */

#ifdef GL_OES_gpu_shader5

static void _glewInfo_GL_OES_gpu_shader5 (void)
{
  glewPrintExt("GL_OES_gpu_shader5", GLEW_OES_gpu_shader5, glewIsSupported("GL_OES_gpu_shader5"), glewGetExtension("GL_OES_gpu_shader5"));
}

#endif /* GL_OES_gpu_shader5 */

#ifdef GL_OES_mapbuffer

static void _glewInfo_GL_OES_mapbuffer (void)
{
  GLboolean fi = glewPrintExt("GL_OES_mapbuffer", GLEW_OES_mapbuffer, glewIsSupported("GL_OES_mapbuffer"), glewGetExtension("GL_OES_mapbuffer"));

  glewInfoFunc(fi, "glGetBufferPointervOES", glGetBufferPointervOES == NULL);
  glewInfoFunc(fi, "glMapBufferOES", glMapBufferOES == NULL);
  glewInfoFunc(fi, "glUnmapBufferOES", glUnmapBufferOES == NULL);
}

#endif /* GL_OES_mapbuffer */

#ifdef GL_OES_matrix_get

static void _glewInfo_GL_OES_matrix_get (void)
{
  glewPrintExt("GL_OES_matrix_get", GLEW_OES_matrix_get, glewIsSupported("GL_OES_matrix_get"), glewGetExtension("GL_OES_matrix_get"));
}

#endif /* GL_OES_matrix_get */

#ifdef GL_OES_matrix_palette

static void _glewInfo_GL_OES_matrix_palette (void)
{
  GLboolean fi = glewPrintExt("GL_OES_matrix_palette", GLEW_OES_matrix_palette, glewIsSupported("GL_OES_matrix_palette"), glewGetExtension("GL_OES_matrix_palette"));

  glewInfoFunc(fi, "glCurrentPaletteMatrixOES", glCurrentPaletteMatrixOES == NULL);
  glewInfoFunc(fi, "glMatrixIndexPointerOES", glMatrixIndexPointerOES == NULL);
  glewInfoFunc(fi, "glWeightPointerOES", glWeightPointerOES == NULL);
}

#endif /* GL_OES_matrix_palette */

#ifdef GL_OES_packed_depth_stencil

static void _glewInfo_GL_OES_packed_depth_stencil (void)
{
  glewPrintExt("GL_OES_packed_depth_stencil", GLEW_OES_packed_depth_stencil, glewIsSupported("GL_OES_packed_depth_stencil"), glewGetExtension("GL_OES_packed_depth_stencil"));
}

#endif /* GL_OES_packed_depth_stencil */

#ifdef GL_OES_point_size_array

static void _glewInfo_GL_OES_point_size_array (void)
{
  glewPrintExt("GL_OES_point_size_array", GLEW_OES_point_size_array, glewIsSupported("GL_OES_point_size_array"), glewGetExtension("GL_OES_point_size_array"));
}

#endif /* GL_OES_point_size_array */

#ifdef GL_OES_point_sprite

static void _glewInfo_GL_OES_point_sprite (void)
{
  glewPrintExt("GL_OES_point_sprite", GLEW_OES_point_sprite, glewIsSupported("GL_OES_point_sprite"), glewGetExtension("GL_OES_point_sprite"));
}

#endif /* GL_OES_point_sprite */

#ifdef GL_OES_read_format

static void _glewInfo_GL_OES_read_format (void)
{
  glewPrintExt("GL_OES_read_format", GLEW_OES_read_format, glewIsSupported("GL_OES_read_format"), glewGetExtension("GL_OES_read_format"));
}

#endif /* GL_OES_read_format */

#ifdef GL_OES_required_internalformat

static void _glewInfo_GL_OES_required_internalformat (void)
{
  glewPrintExt("GL_OES_required_internalformat", GLEW_OES_required_internalformat, glewIsSupported("GL_OES_required_internalformat"), glewGetExtension("GL_OES_required_internalformat"));
}

#endif /* GL_OES_required_internalformat */

#ifdef GL_OES_rgb8_rgba8

static void _glewInfo_GL_OES_rgb8_rgba8 (void)
{
  glewPrintExt("GL_OES_rgb8_rgba8", GLEW_OES_rgb8_rgba8, glewIsSupported("GL_OES_rgb8_rgba8"), glewGetExtension("GL_OES_rgb8_rgba8"));
}

#endif /* GL_OES_rgb8_rgba8 */

#ifdef GL_OES_sample_shading

static void _glewInfo_GL_OES_sample_shading (void)
{
  GLboolean fi = glewPrintExt("GL_OES_sample_shading", GLEW_OES_sample_shading, glewIsSupported("GL_OES_sample_shading"), glewGetExtension("GL_OES_sample_shading"));

  glewInfoFunc(fi, "glMinSampleShadingOES", glMinSampleShadingOES == NULL);
}

#endif /* GL_OES_sample_shading */

#ifdef GL_OES_sample_variables

static void _glewInfo_GL_OES_sample_variables (void)
{
  glewPrintExt("GL_OES_sample_variables", GLEW_OES_sample_variables, glewIsSupported("GL_OES_sample_variables"), glewGetExtension("GL_OES_sample_variables"));
}

#endif /* GL_OES_sample_variables */

#ifdef GL_OES_shader_image_atomic

static void _glewInfo_GL_OES_shader_image_atomic (void)
{
  glewPrintExt("GL_OES_shader_image_atomic", GLEW_OES_shader_image_atomic, glewIsSupported("GL_OES_shader_image_atomic"), glewGetExtension("GL_OES_shader_image_atomic"));
}

#endif /* GL_OES_shader_image_atomic */

#ifdef GL_OES_shader_io_blocks

static void _glewInfo_GL_OES_shader_io_blocks (void)
{
  glewPrintExt("GL_OES_shader_io_blocks", GLEW_OES_shader_io_blocks, glewIsSupported("GL_OES_shader_io_blocks"), glewGetExtension("GL_OES_shader_io_blocks"));
}

#endif /* GL_OES_shader_io_blocks */

#ifdef GL_OES_shader_multisample_interpolation

static void _glewInfo_GL_OES_shader_multisample_interpolation (void)
{
  glewPrintExt("GL_OES_shader_multisample_interpolation", GLEW_OES_shader_multisample_interpolation, glewIsSupported("GL_OES_shader_multisample_interpolation"), glewGetExtension("GL_OES_shader_multisample_interpolation"));
}

#endif /* GL_OES_shader_multisample_interpolation */

#ifdef GL_OES_single_precision

static void _glewInfo_GL_OES_single_precision (void)
{
  GLboolean fi = glewPrintExt("GL_OES_single_precision", GLEW_OES_single_precision, glewIsSupported("GL_OES_single_precision"), glewGetExtension("GL_OES_single_precision"));

  glewInfoFunc(fi, "glClearDepthfOES", glClearDepthfOES == NULL);
  glewInfoFunc(fi, "glClipPlanefOES", glClipPlanefOES == NULL);
  glewInfoFunc(fi, "glDepthRangefOES", glDepthRangefOES == NULL);
  glewInfoFunc(fi, "glFrustumfOES", glFrustumfOES == NULL);
  glewInfoFunc(fi, "glGetClipPlanefOES", glGetClipPlanefOES == NULL);
  glewInfoFunc(fi, "glOrthofOES", glOrthofOES == NULL);
}

#endif /* GL_OES_single_precision */

#ifdef GL_OES_standard_derivatives

static void _glewInfo_GL_OES_standard_derivatives (void)
{
  glewPrintExt("GL_OES_standard_derivatives", GLEW_OES_standard_derivatives, glewIsSupported("GL_OES_standard_derivatives"), glewGetExtension("GL_OES_standard_derivatives"));
}

#endif /* GL_OES_standard_derivatives */

#ifdef GL_OES_stencil1

static void _glewInfo_GL_OES_stencil1 (void)
{
  glewPrintExt("GL_OES_stencil1", GLEW_OES_stencil1, glewIsSupported("GL_OES_stencil1"), glewGetExtension("GL_OES_stencil1"));
}

#endif /* GL_OES_stencil1 */

#ifdef GL_OES_stencil4

static void _glewInfo_GL_OES_stencil4 (void)
{
  glewPrintExt("GL_OES_stencil4", GLEW_OES_stencil4, glewIsSupported("GL_OES_stencil4"), glewGetExtension("GL_OES_stencil4"));
}

#endif /* GL_OES_stencil4 */

#ifdef GL_OES_stencil8

static void _glewInfo_GL_OES_stencil8 (void)
{
  glewPrintExt("GL_OES_stencil8", GLEW_OES_stencil8, glewIsSupported("GL_OES_stencil8"), glewGetExtension("GL_OES_stencil8"));
}

#endif /* GL_OES_stencil8 */

#ifdef GL_OES_surfaceless_context

static void _glewInfo_GL_OES_surfaceless_context (void)
{
  glewPrintExt("GL_OES_surfaceless_context", GLEW_OES_surfaceless_context, glewIsSupported("GL_OES_surfaceless_context"), glewGetExtension("GL_OES_surfaceless_context"));
}

#endif /* GL_OES_surfaceless_context */

#ifdef GL_OES_tessellation_point_size

static void _glewInfo_GL_OES_tessellation_point_size (void)
{
  glewPrintExt("GL_OES_tessellation_point_size", GLEW_OES_tessellation_point_size, glewIsSupported("GL_OES_tessellation_point_size"), glewGetExtension("GL_OES_tessellation_point_size"));
}

#endif /* GL_OES_tessellation_point_size */

#ifdef GL_OES_tessellation_shader

static void _glewInfo_GL_OES_tessellation_shader (void)
{
  glewPrintExt("GL_OES_tessellation_shader", GLEW_OES_tessellation_shader, glewIsSupported("GL_OES_tessellation_shader"), glewGetExtension("GL_OES_tessellation_shader"));
}

#endif /* GL_OES_tessellation_shader */

#ifdef GL_OES_texture_3D

static void _glewInfo_GL_OES_texture_3D (void)
{
  GLboolean fi = glewPrintExt("GL_OES_texture_3D", GLEW_OES_texture_3D, glewIsSupported("GL_OES_texture_3D"), glewGetExtension("GL_OES_texture_3D"));

  glewInfoFunc(fi, "glCompressedTexImage3DOES", glCompressedTexImage3DOES == NULL);
  glewInfoFunc(fi, "glCompressedTexSubImage3DOES", glCompressedTexSubImage3DOES == NULL);
  glewInfoFunc(fi, "glCopyTexSubImage3DOES", glCopyTexSubImage3DOES == NULL);
  glewInfoFunc(fi, "glFramebufferTexture3DOES", glFramebufferTexture3DOES == NULL);
  glewInfoFunc(fi, "glTexImage3DOES", glTexImage3DOES == NULL);
  glewInfoFunc(fi, "glTexSubImage3DOES", glTexSubImage3DOES == NULL);
}

#endif /* GL_OES_texture_3D */

#ifdef GL_OES_texture_border_clamp

static void _glewInfo_GL_OES_texture_border_clamp (void)
{
  GLboolean fi = glewPrintExt("GL_OES_texture_border_clamp", GLEW_OES_texture_border_clamp, glewIsSupported("GL_OES_texture_border_clamp"), glewGetExtension("GL_OES_texture_border_clamp"));

  glewInfoFunc(fi, "glGetSamplerParameterIivOES", glGetSamplerParameterIivOES == NULL);
  glewInfoFunc(fi, "glGetSamplerParameterIuivOES", glGetSamplerParameterIuivOES == NULL);
  glewInfoFunc(fi, "glGetTexParameterIivOES", glGetTexParameterIivOES == NULL);
  glewInfoFunc(fi, "glGetTexParameterIuivOES", glGetTexParameterIuivOES == NULL);
  glewInfoFunc(fi, "glSamplerParameterIivOES", glSamplerParameterIivOES == NULL);
  glewInfoFunc(fi, "glSamplerParameterIuivOES", glSamplerParameterIuivOES == NULL);
  glewInfoFunc(fi, "glTexParameterIivOES", glTexParameterIivOES == NULL);
  glewInfoFunc(fi, "glTexParameterIuivOES", glTexParameterIuivOES == NULL);
}

#endif /* GL_OES_texture_border_clamp */

#ifdef GL_OES_texture_buffer

static void _glewInfo_GL_OES_texture_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_OES_texture_buffer", GLEW_OES_texture_buffer, glewIsSupported("GL_OES_texture_buffer"), glewGetExtension("GL_OES_texture_buffer"));

  glewInfoFunc(fi, "glTexBufferOES", glTexBufferOES == NULL);
  glewInfoFunc(fi, "glTexBufferRangeOES", glTexBufferRangeOES == NULL);
}

#endif /* GL_OES_texture_buffer */

#ifdef GL_OES_texture_compression_astc

static void _glewInfo_GL_OES_texture_compression_astc (void)
{
  glewPrintExt("GL_OES_texture_compression_astc", GLEW_OES_texture_compression_astc, glewIsSupported("GL_OES_texture_compression_astc"), glewGetExtension("GL_OES_texture_compression_astc"));
}

#endif /* GL_OES_texture_compression_astc */

#ifdef GL_OES_texture_cube_map

static void _glewInfo_GL_OES_texture_cube_map (void)
{
  GLboolean fi = glewPrintExt("GL_OES_texture_cube_map", GLEW_OES_texture_cube_map, glewIsSupported("GL_OES_texture_cube_map"), glewGetExtension("GL_OES_texture_cube_map"));

  glewInfoFunc(fi, "glGetTexGenfvOES", glGetTexGenfvOES == NULL);
  glewInfoFunc(fi, "glGetTexGenivOES", glGetTexGenivOES == NULL);
  glewInfoFunc(fi, "glGetTexGenxvOES", glGetTexGenxvOES == NULL);
  glewInfoFunc(fi, "glTexGenfOES", glTexGenfOES == NULL);
  glewInfoFunc(fi, "glTexGenfvOES", glTexGenfvOES == NULL);
  glewInfoFunc(fi, "glTexGeniOES", glTexGeniOES == NULL);
  glewInfoFunc(fi, "glTexGenivOES", glTexGenivOES == NULL);
  glewInfoFunc(fi, "glTexGenxOES", glTexGenxOES == NULL);
  glewInfoFunc(fi, "glTexGenxvOES", glTexGenxvOES == NULL);
}

#endif /* GL_OES_texture_cube_map */

#ifdef GL_OES_texture_cube_map_array

static void _glewInfo_GL_OES_texture_cube_map_array (void)
{
  glewPrintExt("GL_OES_texture_cube_map_array", GLEW_OES_texture_cube_map_array, glewIsSupported("GL_OES_texture_cube_map_array"), glewGetExtension("GL_OES_texture_cube_map_array"));
}

#endif /* GL_OES_texture_cube_map_array */

#ifdef GL_OES_texture_env_crossbar

static void _glewInfo_GL_OES_texture_env_crossbar (void)
{
  glewPrintExt("GL_OES_texture_env_crossbar", GLEW_OES_texture_env_crossbar, glewIsSupported("GL_OES_texture_env_crossbar"), glewGetExtension("GL_OES_texture_env_crossbar"));
}

#endif /* GL_OES_texture_env_crossbar */

#ifdef GL_OES_texture_mirrored_repeat

static void _glewInfo_GL_OES_texture_mirrored_repeat (void)
{
  glewPrintExt("GL_OES_texture_mirrored_repeat", GLEW_OES_texture_mirrored_repeat, glewIsSupported("GL_OES_texture_mirrored_repeat"), glewGetExtension("GL_OES_texture_mirrored_repeat"));
}

#endif /* GL_OES_texture_mirrored_repeat */

#ifdef GL_OES_texture_npot

static void _glewInfo_GL_OES_texture_npot (void)
{
  glewPrintExt("GL_OES_texture_npot", GLEW_OES_texture_npot, glewIsSupported("GL_OES_texture_npot"), glewGetExtension("GL_OES_texture_npot"));
}

#endif /* GL_OES_texture_npot */

#ifdef GL_OES_texture_stencil8

static void _glewInfo_GL_OES_texture_stencil8 (void)
{
  glewPrintExt("GL_OES_texture_stencil8", GLEW_OES_texture_stencil8, glewIsSupported("GL_OES_texture_stencil8"), glewGetExtension("GL_OES_texture_stencil8"));
}

#endif /* GL_OES_texture_stencil8 */

#ifdef GL_OES_texture_storage_multisample_2d_array

static void _glewInfo_GL_OES_texture_storage_multisample_2d_array (void)
{
  GLboolean fi = glewPrintExt("GL_OES_texture_storage_multisample_2d_array", GLEW_OES_texture_storage_multisample_2d_array, glewIsSupported("GL_OES_texture_storage_multisample_2d_array"), glewGetExtension("GL_OES_texture_storage_multisample_2d_array"));

  glewInfoFunc(fi, "glTexStorage3DMultisampleOES", glTexStorage3DMultisampleOES == NULL);
}

#endif /* GL_OES_texture_storage_multisample_2d_array */

#ifdef GL_OES_texture_view

static void _glewInfo_GL_OES_texture_view (void)
{
  GLboolean fi = glewPrintExt("GL_OES_texture_view", GLEW_OES_texture_view, glewIsSupported("GL_OES_texture_view"), glewGetExtension("GL_OES_texture_view"));

  glewInfoFunc(fi, "glTextureViewOES", glTextureViewOES == NULL);
}

#endif /* GL_OES_texture_view */

#ifdef GL_OES_vertex_array_object

static void _glewInfo_GL_OES_vertex_array_object (void)
{
  GLboolean fi = glewPrintExt("GL_OES_vertex_array_object", GLEW_OES_vertex_array_object, glewIsSupported("GL_OES_vertex_array_object"), glewGetExtension("GL_OES_vertex_array_object"));

  glewInfoFunc(fi, "glBindVertexArrayOES", glBindVertexArrayOES == NULL);
  glewInfoFunc(fi, "glDeleteVertexArraysOES", glDeleteVertexArraysOES == NULL);
  glewInfoFunc(fi, "glGenVertexArraysOES", glGenVertexArraysOES == NULL);
  glewInfoFunc(fi, "glIsVertexArrayOES", glIsVertexArrayOES == NULL);
}

#endif /* GL_OES_vertex_array_object */

#ifdef GL_OES_vertex_half_float

static void _glewInfo_GL_OES_vertex_half_float (void)
{
  glewPrintExt("GL_OES_vertex_half_float", GLEW_OES_vertex_half_float, glewIsSupported("GL_OES_vertex_half_float"), glewGetExtension("GL_OES_vertex_half_float"));
}

#endif /* GL_OES_vertex_half_float */

#ifdef GL_OES_vertex_type_10_10_10_2

static void _glewInfo_GL_OES_vertex_type_10_10_10_2 (void)
{
  glewPrintExt("GL_OES_vertex_type_10_10_10_2", GLEW_OES_vertex_type_10_10_10_2, glewIsSupported("GL_OES_vertex_type_10_10_10_2"), glewGetExtension("GL_OES_vertex_type_10_10_10_2"));
}

#endif /* GL_OES_vertex_type_10_10_10_2 */

#ifdef GL_OML_interlace

static void _glewInfo_GL_OML_interlace (void)
{
  glewPrintExt("GL_OML_interlace", GLEW_OML_interlace, glewIsSupported("GL_OML_interlace"), glewGetExtension("GL_OML_interlace"));
}

#endif /* GL_OML_interlace */

#ifdef GL_OML_resample

static void _glewInfo_GL_OML_resample (void)
{
  glewPrintExt("GL_OML_resample", GLEW_OML_resample, glewIsSupported("GL_OML_resample"), glewGetExtension("GL_OML_resample"));
}

#endif /* GL_OML_resample */

#ifdef GL_OML_subsample

static void _glewInfo_GL_OML_subsample (void)
{
  glewPrintExt("GL_OML_subsample", GLEW_OML_subsample, glewIsSupported("GL_OML_subsample"), glewGetExtension("GL_OML_subsample"));
}

#endif /* GL_OML_subsample */

#ifdef GL_OVR_multiview

static void _glewInfo_GL_OVR_multiview (void)
{
  GLboolean fi = glewPrintExt("GL_OVR_multiview", GLEW_OVR_multiview, glewIsSupported("GL_OVR_multiview"), glewGetExtension("GL_OVR_multiview"));

  glewInfoFunc(fi, "glFramebufferTextureMultiviewOVR", glFramebufferTextureMultiviewOVR == NULL);
  glewInfoFunc(fi, "glNamedFramebufferTextureMultiviewOVR", glNamedFramebufferTextureMultiviewOVR == NULL);
}

#endif /* GL_OVR_multiview */

#ifdef GL_OVR_multiview2

static void _glewInfo_GL_OVR_multiview2 (void)
{
  glewPrintExt("GL_OVR_multiview2", GLEW_OVR_multiview2, glewIsSupported("GL_OVR_multiview2"), glewGetExtension("GL_OVR_multiview2"));
}

#endif /* GL_OVR_multiview2 */

#ifdef GL_OVR_multiview_multisampled_render_to_texture

static void _glewInfo_GL_OVR_multiview_multisampled_render_to_texture (void)
{
  GLboolean fi = glewPrintExt("GL_OVR_multiview_multisampled_render_to_texture", GLEW_OVR_multiview_multisampled_render_to_texture, glewIsSupported("GL_OVR_multiview_multisampled_render_to_texture"), glewGetExtension("GL_OVR_multiview_multisampled_render_to_texture"));

  glewInfoFunc(fi, "glFramebufferTextureMultisampleMultiviewOVR", glFramebufferTextureMultisampleMultiviewOVR == NULL);
}

#endif /* GL_OVR_multiview_multisampled_render_to_texture */

#ifdef GL_PGI_misc_hints

static void _glewInfo_GL_PGI_misc_hints (void)
{
  glewPrintExt("GL_PGI_misc_hints", GLEW_PGI_misc_hints, glewIsSupported("GL_PGI_misc_hints"), glewGetExtension("GL_PGI_misc_hints"));
}

#endif /* GL_PGI_misc_hints */

#ifdef GL_PGI_vertex_hints

static void _glewInfo_GL_PGI_vertex_hints (void)
{
  glewPrintExt("GL_PGI_vertex_hints", GLEW_PGI_vertex_hints, glewIsSupported("GL_PGI_vertex_hints"), glewGetExtension("GL_PGI_vertex_hints"));
}

#endif /* GL_PGI_vertex_hints */

#ifdef GL_QCOM_YUV_texture_gather

static void _glewInfo_GL_QCOM_YUV_texture_gather (void)
{
  glewPrintExt("GL_QCOM_YUV_texture_gather", GLEW_QCOM_YUV_texture_gather, glewIsSupported("GL_QCOM_YUV_texture_gather"), glewGetExtension("GL_QCOM_YUV_texture_gather"));
}

#endif /* GL_QCOM_YUV_texture_gather */

#ifdef GL_QCOM_alpha_test

static void _glewInfo_GL_QCOM_alpha_test (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_alpha_test", GLEW_QCOM_alpha_test, glewIsSupported("GL_QCOM_alpha_test"), glewGetExtension("GL_QCOM_alpha_test"));

  glewInfoFunc(fi, "glAlphaFuncQCOM", glAlphaFuncQCOM == NULL);
}

#endif /* GL_QCOM_alpha_test */

#ifdef GL_QCOM_binning_control

static void _glewInfo_GL_QCOM_binning_control (void)
{
  glewPrintExt("GL_QCOM_binning_control", GLEW_QCOM_binning_control, glewIsSupported("GL_QCOM_binning_control"), glewGetExtension("GL_QCOM_binning_control"));
}

#endif /* GL_QCOM_binning_control */

#ifdef GL_QCOM_driver_control

static void _glewInfo_GL_QCOM_driver_control (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_driver_control", GLEW_QCOM_driver_control, glewIsSupported("GL_QCOM_driver_control"), glewGetExtension("GL_QCOM_driver_control"));

  glewInfoFunc(fi, "glDisableDriverControlQCOM", glDisableDriverControlQCOM == NULL);
  glewInfoFunc(fi, "glEnableDriverControlQCOM", glEnableDriverControlQCOM == NULL);
  glewInfoFunc(fi, "glGetDriverControlStringQCOM", glGetDriverControlStringQCOM == NULL);
  glewInfoFunc(fi, "glGetDriverControlsQCOM", glGetDriverControlsQCOM == NULL);
}

#endif /* GL_QCOM_driver_control */

#ifdef GL_QCOM_extended_get

static void _glewInfo_GL_QCOM_extended_get (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_extended_get", GLEW_QCOM_extended_get, glewIsSupported("GL_QCOM_extended_get"), glewGetExtension("GL_QCOM_extended_get"));

  glewInfoFunc(fi, "glExtGetBufferPointervQCOM", glExtGetBufferPointervQCOM == NULL);
  glewInfoFunc(fi, "glExtGetBuffersQCOM", glExtGetBuffersQCOM == NULL);
  glewInfoFunc(fi, "glExtGetFramebuffersQCOM", glExtGetFramebuffersQCOM == NULL);
  glewInfoFunc(fi, "glExtGetRenderbuffersQCOM", glExtGetRenderbuffersQCOM == NULL);
  glewInfoFunc(fi, "glExtGetTexLevelParameterivQCOM", glExtGetTexLevelParameterivQCOM == NULL);
  glewInfoFunc(fi, "glExtGetTexSubImageQCOM", glExtGetTexSubImageQCOM == NULL);
  glewInfoFunc(fi, "glExtGetTexturesQCOM", glExtGetTexturesQCOM == NULL);
  glewInfoFunc(fi, "glExtTexObjectStateOverrideiQCOM", glExtTexObjectStateOverrideiQCOM == NULL);
}

#endif /* GL_QCOM_extended_get */

#ifdef GL_QCOM_extended_get2

static void _glewInfo_GL_QCOM_extended_get2 (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_extended_get2", GLEW_QCOM_extended_get2, glewIsSupported("GL_QCOM_extended_get2"), glewGetExtension("GL_QCOM_extended_get2"));

  glewInfoFunc(fi, "glExtGetProgramBinarySourceQCOM", glExtGetProgramBinarySourceQCOM == NULL);
  glewInfoFunc(fi, "glExtGetProgramsQCOM", glExtGetProgramsQCOM == NULL);
  glewInfoFunc(fi, "glExtGetShadersQCOM", glExtGetShadersQCOM == NULL);
  glewInfoFunc(fi, "glExtIsProgramBinaryQCOM", glExtIsProgramBinaryQCOM == NULL);
}

#endif /* GL_QCOM_extended_get2 */

#ifdef GL_QCOM_framebuffer_foveated

static void _glewInfo_GL_QCOM_framebuffer_foveated (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_framebuffer_foveated", GLEW_QCOM_framebuffer_foveated, glewIsSupported("GL_QCOM_framebuffer_foveated"), glewGetExtension("GL_QCOM_framebuffer_foveated"));

  glewInfoFunc(fi, "glFramebufferFoveationConfigQCOM", glFramebufferFoveationConfigQCOM == NULL);
  glewInfoFunc(fi, "glFramebufferFoveationParametersQCOM", glFramebufferFoveationParametersQCOM == NULL);
}

#endif /* GL_QCOM_framebuffer_foveated */

#ifdef GL_QCOM_perfmon_global_mode

static void _glewInfo_GL_QCOM_perfmon_global_mode (void)
{
  glewPrintExt("GL_QCOM_perfmon_global_mode", GLEW_QCOM_perfmon_global_mode, glewIsSupported("GL_QCOM_perfmon_global_mode"), glewGetExtension("GL_QCOM_perfmon_global_mode"));
}

#endif /* GL_QCOM_perfmon_global_mode */

#ifdef GL_QCOM_shader_framebuffer_fetch_noncoherent

static void _glewInfo_GL_QCOM_shader_framebuffer_fetch_noncoherent (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_shader_framebuffer_fetch_noncoherent", GLEW_QCOM_shader_framebuffer_fetch_noncoherent, glewIsSupported("GL_QCOM_shader_framebuffer_fetch_noncoherent"), glewGetExtension("GL_QCOM_shader_framebuffer_fetch_noncoherent"));

  glewInfoFunc(fi, "glFramebufferFetchBarrierQCOM", glFramebufferFetchBarrierQCOM == NULL);
}

#endif /* GL_QCOM_shader_framebuffer_fetch_noncoherent */

#ifdef GL_QCOM_shader_framebuffer_fetch_rate

static void _glewInfo_GL_QCOM_shader_framebuffer_fetch_rate (void)
{
  glewPrintExt("GL_QCOM_shader_framebuffer_fetch_rate", GLEW_QCOM_shader_framebuffer_fetch_rate, glewIsSupported("GL_QCOM_shader_framebuffer_fetch_rate"), glewGetExtension("GL_QCOM_shader_framebuffer_fetch_rate"));
}

#endif /* GL_QCOM_shader_framebuffer_fetch_rate */

#ifdef GL_QCOM_texture_foveated

static void _glewInfo_GL_QCOM_texture_foveated (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_texture_foveated", GLEW_QCOM_texture_foveated, glewIsSupported("GL_QCOM_texture_foveated"), glewGetExtension("GL_QCOM_texture_foveated"));

  glewInfoFunc(fi, "glTextureFoveationParametersQCOM", glTextureFoveationParametersQCOM == NULL);
}

#endif /* GL_QCOM_texture_foveated */

#ifdef GL_QCOM_texture_foveated_subsampled_layout

static void _glewInfo_GL_QCOM_texture_foveated_subsampled_layout (void)
{
  glewPrintExt("GL_QCOM_texture_foveated_subsampled_layout", GLEW_QCOM_texture_foveated_subsampled_layout, glewIsSupported("GL_QCOM_texture_foveated_subsampled_layout"), glewGetExtension("GL_QCOM_texture_foveated_subsampled_layout"));
}

#endif /* GL_QCOM_texture_foveated_subsampled_layout */

#ifdef GL_QCOM_tiled_rendering

static void _glewInfo_GL_QCOM_tiled_rendering (void)
{
  GLboolean fi = glewPrintExt("GL_QCOM_tiled_rendering", GLEW_QCOM_tiled_rendering, glewIsSupported("GL_QCOM_tiled_rendering"), glewGetExtension("GL_QCOM_tiled_rendering"));

  glewInfoFunc(fi, "glEndTilingQCOM", glEndTilingQCOM == NULL);
  glewInfoFunc(fi, "glStartTilingQCOM", glStartTilingQCOM == NULL);
}

#endif /* GL_QCOM_tiled_rendering */

#ifdef GL_QCOM_writeonly_rendering

static void _glewInfo_GL_QCOM_writeonly_rendering (void)
{
  glewPrintExt("GL_QCOM_writeonly_rendering", GLEW_QCOM_writeonly_rendering, glewIsSupported("GL_QCOM_writeonly_rendering"), glewGetExtension("GL_QCOM_writeonly_rendering"));
}

#endif /* GL_QCOM_writeonly_rendering */

#ifdef GL_REGAL_ES1_0_compatibility

static void _glewInfo_GL_REGAL_ES1_0_compatibility (void)
{
  GLboolean fi = glewPrintExt("GL_REGAL_ES1_0_compatibility", GLEW_REGAL_ES1_0_compatibility, glewIsSupported("GL_REGAL_ES1_0_compatibility"), glewGetExtension("GL_REGAL_ES1_0_compatibility"));

  glewInfoFunc(fi, "glAlphaFuncx", glAlphaFuncx == NULL);
  glewInfoFunc(fi, "glClearColorx", glClearColorx == NULL);
  glewInfoFunc(fi, "glClearDepthx", glClearDepthx == NULL);
  glewInfoFunc(fi, "glColor4x", glColor4x == NULL);
  glewInfoFunc(fi, "glDepthRangex", glDepthRangex == NULL);
  glewInfoFunc(fi, "glFogx", glFogx == NULL);
  glewInfoFunc(fi, "glFogxv", glFogxv == NULL);
  glewInfoFunc(fi, "glFrustumf", glFrustumf == NULL);
  glewInfoFunc(fi, "glFrustumx", glFrustumx == NULL);
  glewInfoFunc(fi, "glLightModelx", glLightModelx == NULL);
  glewInfoFunc(fi, "glLightModelxv", glLightModelxv == NULL);
  glewInfoFunc(fi, "glLightx", glLightx == NULL);
  glewInfoFunc(fi, "glLightxv", glLightxv == NULL);
  glewInfoFunc(fi, "glLineWidthx", glLineWidthx == NULL);
  glewInfoFunc(fi, "glLoadMatrixx", glLoadMatrixx == NULL);
  glewInfoFunc(fi, "glMaterialx", glMaterialx == NULL);
  glewInfoFunc(fi, "glMaterialxv", glMaterialxv == NULL);
  glewInfoFunc(fi, "glMultMatrixx", glMultMatrixx == NULL);
  glewInfoFunc(fi, "glMultiTexCoord4x", glMultiTexCoord4x == NULL);
  glewInfoFunc(fi, "glNormal3x", glNormal3x == NULL);
  glewInfoFunc(fi, "glOrthof", glOrthof == NULL);
  glewInfoFunc(fi, "glOrthox", glOrthox == NULL);
  glewInfoFunc(fi, "glPointSizex", glPointSizex == NULL);
  glewInfoFunc(fi, "glPolygonOffsetx", glPolygonOffsetx == NULL);
  glewInfoFunc(fi, "glRotatex", glRotatex == NULL);
  glewInfoFunc(fi, "glSampleCoveragex", glSampleCoveragex == NULL);
  glewInfoFunc(fi, "glScalex", glScalex == NULL);
  glewInfoFunc(fi, "glTexEnvx", glTexEnvx == NULL);
  glewInfoFunc(fi, "glTexEnvxv", glTexEnvxv == NULL);
  glewInfoFunc(fi, "glTexParameterx", glTexParameterx == NULL);
  glewInfoFunc(fi, "glTranslatex", glTranslatex == NULL);
}

#endif /* GL_REGAL_ES1_0_compatibility */

#ifdef GL_REGAL_ES1_1_compatibility

static void _glewInfo_GL_REGAL_ES1_1_compatibility (void)
{
  GLboolean fi = glewPrintExt("GL_REGAL_ES1_1_compatibility", GLEW_REGAL_ES1_1_compatibility, glewIsSupported("GL_REGAL_ES1_1_compatibility"), glewGetExtension("GL_REGAL_ES1_1_compatibility"));

  glewInfoFunc(fi, "glClipPlanef", glClipPlanef == NULL);
  glewInfoFunc(fi, "glClipPlanex", glClipPlanex == NULL);
  glewInfoFunc(fi, "glGetClipPlanef", glGetClipPlanef == NULL);
  glewInfoFunc(fi, "glGetClipPlanex", glGetClipPlanex == NULL);
  glewInfoFunc(fi, "glGetFixedv", glGetFixedv == NULL);
  glewInfoFunc(fi, "glGetLightxv", glGetLightxv == NULL);
  glewInfoFunc(fi, "glGetMaterialxv", glGetMaterialxv == NULL);
  glewInfoFunc(fi, "glGetTexEnvxv", glGetTexEnvxv == NULL);
  glewInfoFunc(fi, "glGetTexParameterxv", glGetTexParameterxv == NULL);
  glewInfoFunc(fi, "glPointParameterx", glPointParameterx == NULL);
  glewInfoFunc(fi, "glPointParameterxv", glPointParameterxv == NULL);
  glewInfoFunc(fi, "glPointSizePointerOES", glPointSizePointerOES == NULL);
  glewInfoFunc(fi, "glTexParameterxv", glTexParameterxv == NULL);
}

#endif /* GL_REGAL_ES1_1_compatibility */

#ifdef GL_REGAL_enable

static void _glewInfo_GL_REGAL_enable (void)
{
  glewPrintExt("GL_REGAL_enable", GLEW_REGAL_enable, glewIsSupported("GL_REGAL_enable"), glewGetExtension("GL_REGAL_enable"));
}

#endif /* GL_REGAL_enable */

#ifdef GL_REGAL_error_string

static void _glewInfo_GL_REGAL_error_string (void)
{
  GLboolean fi = glewPrintExt("GL_REGAL_error_string", GLEW_REGAL_error_string, glewIsSupported("GL_REGAL_error_string"), glewGetExtension("GL_REGAL_error_string"));

  glewInfoFunc(fi, "glErrorStringREGAL", glErrorStringREGAL == NULL);
}

#endif /* GL_REGAL_error_string */

#ifdef GL_REGAL_extension_query

static void _glewInfo_GL_REGAL_extension_query (void)
{
  GLboolean fi = glewPrintExt("GL_REGAL_extension_query", GLEW_REGAL_extension_query, glewIsSupported("GL_REGAL_extension_query"), glewGetExtension("GL_REGAL_extension_query"));

  glewInfoFunc(fi, "glGetExtensionREGAL", glGetExtensionREGAL == NULL);
  glewInfoFunc(fi, "glIsSupportedREGAL", glIsSupportedREGAL == NULL);
}

#endif /* GL_REGAL_extension_query */

#ifdef GL_REGAL_log

static void _glewInfo_GL_REGAL_log (void)
{
  GLboolean fi = glewPrintExt("GL_REGAL_log", GLEW_REGAL_log, glewIsSupported("GL_REGAL_log"), glewGetExtension("GL_REGAL_log"));

  glewInfoFunc(fi, "glLogMessageCallbackREGAL", glLogMessageCallbackREGAL == NULL);
}

#endif /* GL_REGAL_log */

#ifdef GL_REGAL_proc_address

static void _glewInfo_GL_REGAL_proc_address (void)
{
  GLboolean fi = glewPrintExt("GL_REGAL_proc_address", GLEW_REGAL_proc_address, glewIsSupported("GL_REGAL_proc_address"), glewGetExtension("GL_REGAL_proc_address"));

  glewInfoFunc(fi, "glGetProcAddressREGAL", glGetProcAddressREGAL == NULL);
}

#endif /* GL_REGAL_proc_address */

#ifdef GL_REND_screen_coordinates

static void _glewInfo_GL_REND_screen_coordinates (void)
{
  glewPrintExt("GL_REND_screen_coordinates", GLEW_REND_screen_coordinates, glewIsSupported("GL_REND_screen_coordinates"), glewGetExtension("GL_REND_screen_coordinates"));
}

#endif /* GL_REND_screen_coordinates */

#ifdef GL_S3_s3tc

static void _glewInfo_GL_S3_s3tc (void)
{
  glewPrintExt("GL_S3_s3tc", GLEW_S3_s3tc, glewIsSupported("GL_S3_s3tc"), glewGetExtension("GL_S3_s3tc"));
}

#endif /* GL_S3_s3tc */

#ifdef GL_SGIS_clip_band_hint

static void _glewInfo_GL_SGIS_clip_band_hint (void)
{
  glewPrintExt("GL_SGIS_clip_band_hint", GLEW_SGIS_clip_band_hint, glewIsSupported("GL_SGIS_clip_band_hint"), glewGetExtension("GL_SGIS_clip_band_hint"));
}

#endif /* GL_SGIS_clip_band_hint */

#ifdef GL_SGIS_color_range

static void _glewInfo_GL_SGIS_color_range (void)
{
  glewPrintExt("GL_SGIS_color_range", GLEW_SGIS_color_range, glewIsSupported("GL_SGIS_color_range"), glewGetExtension("GL_SGIS_color_range"));
}

#endif /* GL_SGIS_color_range */

#ifdef GL_SGIS_detail_texture

static void _glewInfo_GL_SGIS_detail_texture (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_detail_texture", GLEW_SGIS_detail_texture, glewIsSupported("GL_SGIS_detail_texture"), glewGetExtension("GL_SGIS_detail_texture"));

  glewInfoFunc(fi, "glDetailTexFuncSGIS", glDetailTexFuncSGIS == NULL);
  glewInfoFunc(fi, "glGetDetailTexFuncSGIS", glGetDetailTexFuncSGIS == NULL);
}

#endif /* GL_SGIS_detail_texture */

#ifdef GL_SGIS_fog_function

static void _glewInfo_GL_SGIS_fog_function (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_fog_function", GLEW_SGIS_fog_function, glewIsSupported("GL_SGIS_fog_function"), glewGetExtension("GL_SGIS_fog_function"));

  glewInfoFunc(fi, "glFogFuncSGIS", glFogFuncSGIS == NULL);
  glewInfoFunc(fi, "glGetFogFuncSGIS", glGetFogFuncSGIS == NULL);
}

#endif /* GL_SGIS_fog_function */

#ifdef GL_SGIS_generate_mipmap

static void _glewInfo_GL_SGIS_generate_mipmap (void)
{
  glewPrintExt("GL_SGIS_generate_mipmap", GLEW_SGIS_generate_mipmap, glewIsSupported("GL_SGIS_generate_mipmap"), glewGetExtension("GL_SGIS_generate_mipmap"));
}

#endif /* GL_SGIS_generate_mipmap */

#ifdef GL_SGIS_line_texgen

static void _glewInfo_GL_SGIS_line_texgen (void)
{
  glewPrintExt("GL_SGIS_line_texgen", GLEW_SGIS_line_texgen, glewIsSupported("GL_SGIS_line_texgen"), glewGetExtension("GL_SGIS_line_texgen"));
}

#endif /* GL_SGIS_line_texgen */

#ifdef GL_SGIS_multisample

static void _glewInfo_GL_SGIS_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_multisample", GLEW_SGIS_multisample, glewIsSupported("GL_SGIS_multisample"), glewGetExtension("GL_SGIS_multisample"));

  glewInfoFunc(fi, "glSampleMaskSGIS", glSampleMaskSGIS == NULL);
  glewInfoFunc(fi, "glSamplePatternSGIS", glSamplePatternSGIS == NULL);
}

#endif /* GL_SGIS_multisample */

#ifdef GL_SGIS_multitexture

static void _glewInfo_GL_SGIS_multitexture (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_multitexture", GLEW_SGIS_multitexture, glewIsSupported("GL_SGIS_multitexture"), glewGetExtension("GL_SGIS_multitexture"));

  glewInfoFunc(fi, "glInterleavedTextureCoordSetsSGIS", glInterleavedTextureCoordSetsSGIS == NULL);
  glewInfoFunc(fi, "glSelectTextureCoordSetSGIS", glSelectTextureCoordSetSGIS == NULL);
  glewInfoFunc(fi, "glSelectTextureSGIS", glSelectTextureSGIS == NULL);
  glewInfoFunc(fi, "glSelectTextureTransformSGIS", glSelectTextureTransformSGIS == NULL);
}

#endif /* GL_SGIS_multitexture */

#ifdef GL_SGIS_pixel_texture

static void _glewInfo_GL_SGIS_pixel_texture (void)
{
  glewPrintExt("GL_SGIS_pixel_texture", GLEW_SGIS_pixel_texture, glewIsSupported("GL_SGIS_pixel_texture"), glewGetExtension("GL_SGIS_pixel_texture"));
}

#endif /* GL_SGIS_pixel_texture */

#ifdef GL_SGIS_point_line_texgen

static void _glewInfo_GL_SGIS_point_line_texgen (void)
{
  glewPrintExt("GL_SGIS_point_line_texgen", GLEW_SGIS_point_line_texgen, glewIsSupported("GL_SGIS_point_line_texgen"), glewGetExtension("GL_SGIS_point_line_texgen"));
}

#endif /* GL_SGIS_point_line_texgen */

#ifdef GL_SGIS_shared_multisample

static void _glewInfo_GL_SGIS_shared_multisample (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_shared_multisample", GLEW_SGIS_shared_multisample, glewIsSupported("GL_SGIS_shared_multisample"), glewGetExtension("GL_SGIS_shared_multisample"));

  glewInfoFunc(fi, "glMultisampleSubRectPosSGIS", glMultisampleSubRectPosSGIS == NULL);
}

#endif /* GL_SGIS_shared_multisample */

#ifdef GL_SGIS_sharpen_texture

static void _glewInfo_GL_SGIS_sharpen_texture (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_sharpen_texture", GLEW_SGIS_sharpen_texture, glewIsSupported("GL_SGIS_sharpen_texture"), glewGetExtension("GL_SGIS_sharpen_texture"));

  glewInfoFunc(fi, "glGetSharpenTexFuncSGIS", glGetSharpenTexFuncSGIS == NULL);
  glewInfoFunc(fi, "glSharpenTexFuncSGIS", glSharpenTexFuncSGIS == NULL);
}

#endif /* GL_SGIS_sharpen_texture */

#ifdef GL_SGIS_texture4D

static void _glewInfo_GL_SGIS_texture4D (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_texture4D", GLEW_SGIS_texture4D, glewIsSupported("GL_SGIS_texture4D"), glewGetExtension("GL_SGIS_texture4D"));

  glewInfoFunc(fi, "glTexImage4DSGIS", glTexImage4DSGIS == NULL);
  glewInfoFunc(fi, "glTexSubImage4DSGIS", glTexSubImage4DSGIS == NULL);
}

#endif /* GL_SGIS_texture4D */

#ifdef GL_SGIS_texture_border_clamp

static void _glewInfo_GL_SGIS_texture_border_clamp (void)
{
  glewPrintExt("GL_SGIS_texture_border_clamp", GLEW_SGIS_texture_border_clamp, glewIsSupported("GL_SGIS_texture_border_clamp"), glewGetExtension("GL_SGIS_texture_border_clamp"));
}

#endif /* GL_SGIS_texture_border_clamp */

#ifdef GL_SGIS_texture_edge_clamp

static void _glewInfo_GL_SGIS_texture_edge_clamp (void)
{
  glewPrintExt("GL_SGIS_texture_edge_clamp", GLEW_SGIS_texture_edge_clamp, glewIsSupported("GL_SGIS_texture_edge_clamp"), glewGetExtension("GL_SGIS_texture_edge_clamp"));
}

#endif /* GL_SGIS_texture_edge_clamp */

#ifdef GL_SGIS_texture_filter4

static void _glewInfo_GL_SGIS_texture_filter4 (void)
{
  GLboolean fi = glewPrintExt("GL_SGIS_texture_filter4", GLEW_SGIS_texture_filter4, glewIsSupported("GL_SGIS_texture_filter4"), glewGetExtension("GL_SGIS_texture_filter4"));

  glewInfoFunc(fi, "glGetTexFilterFuncSGIS", glGetTexFilterFuncSGIS == NULL);
  glewInfoFunc(fi, "glTexFilterFuncSGIS", glTexFilterFuncSGIS == NULL);
}

#endif /* GL_SGIS_texture_filter4 */

#ifdef GL_SGIS_texture_lod

static void _glewInfo_GL_SGIS_texture_lod (void)
{
  glewPrintExt("GL_SGIS_texture_lod", GLEW_SGIS_texture_lod, glewIsSupported("GL_SGIS_texture_lod"), glewGetExtension("GL_SGIS_texture_lod"));
}

#endif /* GL_SGIS_texture_lod */

#ifdef GL_SGIS_texture_select

static void _glewInfo_GL_SGIS_texture_select (void)
{
  glewPrintExt("GL_SGIS_texture_select", GLEW_SGIS_texture_select, glewIsSupported("GL_SGIS_texture_select"), glewGetExtension("GL_SGIS_texture_select"));
}

#endif /* GL_SGIS_texture_select */

#ifdef GL_SGIX_async

static void _glewInfo_GL_SGIX_async (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_async", GLEW_SGIX_async, glewIsSupported("GL_SGIX_async"), glewGetExtension("GL_SGIX_async"));

  glewInfoFunc(fi, "glAsyncMarkerSGIX", glAsyncMarkerSGIX == NULL);
  glewInfoFunc(fi, "glDeleteAsyncMarkersSGIX", glDeleteAsyncMarkersSGIX == NULL);
  glewInfoFunc(fi, "glFinishAsyncSGIX", glFinishAsyncSGIX == NULL);
  glewInfoFunc(fi, "glGenAsyncMarkersSGIX", glGenAsyncMarkersSGIX == NULL);
  glewInfoFunc(fi, "glIsAsyncMarkerSGIX", glIsAsyncMarkerSGIX == NULL);
  glewInfoFunc(fi, "glPollAsyncSGIX", glPollAsyncSGIX == NULL);
}

#endif /* GL_SGIX_async */

#ifdef GL_SGIX_async_histogram

static void _glewInfo_GL_SGIX_async_histogram (void)
{
  glewPrintExt("GL_SGIX_async_histogram", GLEW_SGIX_async_histogram, glewIsSupported("GL_SGIX_async_histogram"), glewGetExtension("GL_SGIX_async_histogram"));
}

#endif /* GL_SGIX_async_histogram */

#ifdef GL_SGIX_async_pixel

static void _glewInfo_GL_SGIX_async_pixel (void)
{
  glewPrintExt("GL_SGIX_async_pixel", GLEW_SGIX_async_pixel, glewIsSupported("GL_SGIX_async_pixel"), glewGetExtension("GL_SGIX_async_pixel"));
}

#endif /* GL_SGIX_async_pixel */

#ifdef GL_SGIX_bali_g_instruments

static void _glewInfo_GL_SGIX_bali_g_instruments (void)
{
  glewPrintExt("GL_SGIX_bali_g_instruments", GLEW_SGIX_bali_g_instruments, glewIsSupported("GL_SGIX_bali_g_instruments"), glewGetExtension("GL_SGIX_bali_g_instruments"));
}

#endif /* GL_SGIX_bali_g_instruments */

#ifdef GL_SGIX_bali_r_instruments

static void _glewInfo_GL_SGIX_bali_r_instruments (void)
{
  glewPrintExt("GL_SGIX_bali_r_instruments", GLEW_SGIX_bali_r_instruments, glewIsSupported("GL_SGIX_bali_r_instruments"), glewGetExtension("GL_SGIX_bali_r_instruments"));
}

#endif /* GL_SGIX_bali_r_instruments */

#ifdef GL_SGIX_bali_timer_instruments

static void _glewInfo_GL_SGIX_bali_timer_instruments (void)
{
  glewPrintExt("GL_SGIX_bali_timer_instruments", GLEW_SGIX_bali_timer_instruments, glewIsSupported("GL_SGIX_bali_timer_instruments"), glewGetExtension("GL_SGIX_bali_timer_instruments"));
}

#endif /* GL_SGIX_bali_timer_instruments */

#ifdef GL_SGIX_blend_alpha_minmax

static void _glewInfo_GL_SGIX_blend_alpha_minmax (void)
{
  glewPrintExt("GL_SGIX_blend_alpha_minmax", GLEW_SGIX_blend_alpha_minmax, glewIsSupported("GL_SGIX_blend_alpha_minmax"), glewGetExtension("GL_SGIX_blend_alpha_minmax"));
}

#endif /* GL_SGIX_blend_alpha_minmax */

#ifdef GL_SGIX_blend_cadd

static void _glewInfo_GL_SGIX_blend_cadd (void)
{
  glewPrintExt("GL_SGIX_blend_cadd", GLEW_SGIX_blend_cadd, glewIsSupported("GL_SGIX_blend_cadd"), glewGetExtension("GL_SGIX_blend_cadd"));
}

#endif /* GL_SGIX_blend_cadd */

#ifdef GL_SGIX_blend_cmultiply

static void _glewInfo_GL_SGIX_blend_cmultiply (void)
{
  glewPrintExt("GL_SGIX_blend_cmultiply", GLEW_SGIX_blend_cmultiply, glewIsSupported("GL_SGIX_blend_cmultiply"), glewGetExtension("GL_SGIX_blend_cmultiply"));
}

#endif /* GL_SGIX_blend_cmultiply */

#ifdef GL_SGIX_calligraphic_fragment

static void _glewInfo_GL_SGIX_calligraphic_fragment (void)
{
  glewPrintExt("GL_SGIX_calligraphic_fragment", GLEW_SGIX_calligraphic_fragment, glewIsSupported("GL_SGIX_calligraphic_fragment"), glewGetExtension("GL_SGIX_calligraphic_fragment"));
}

#endif /* GL_SGIX_calligraphic_fragment */

#ifdef GL_SGIX_clipmap

static void _glewInfo_GL_SGIX_clipmap (void)
{
  glewPrintExt("GL_SGIX_clipmap", GLEW_SGIX_clipmap, glewIsSupported("GL_SGIX_clipmap"), glewGetExtension("GL_SGIX_clipmap"));
}

#endif /* GL_SGIX_clipmap */

#ifdef GL_SGIX_color_matrix_accuracy

static void _glewInfo_GL_SGIX_color_matrix_accuracy (void)
{
  glewPrintExt("GL_SGIX_color_matrix_accuracy", GLEW_SGIX_color_matrix_accuracy, glewIsSupported("GL_SGIX_color_matrix_accuracy"), glewGetExtension("GL_SGIX_color_matrix_accuracy"));
}

#endif /* GL_SGIX_color_matrix_accuracy */

#ifdef GL_SGIX_color_table_index_mode

static void _glewInfo_GL_SGIX_color_table_index_mode (void)
{
  glewPrintExt("GL_SGIX_color_table_index_mode", GLEW_SGIX_color_table_index_mode, glewIsSupported("GL_SGIX_color_table_index_mode"), glewGetExtension("GL_SGIX_color_table_index_mode"));
}

#endif /* GL_SGIX_color_table_index_mode */

#ifdef GL_SGIX_complex_polar

static void _glewInfo_GL_SGIX_complex_polar (void)
{
  glewPrintExt("GL_SGIX_complex_polar", GLEW_SGIX_complex_polar, glewIsSupported("GL_SGIX_complex_polar"), glewGetExtension("GL_SGIX_complex_polar"));
}

#endif /* GL_SGIX_complex_polar */

#ifdef GL_SGIX_convolution_accuracy

static void _glewInfo_GL_SGIX_convolution_accuracy (void)
{
  glewPrintExt("GL_SGIX_convolution_accuracy", GLEW_SGIX_convolution_accuracy, glewIsSupported("GL_SGIX_convolution_accuracy"), glewGetExtension("GL_SGIX_convolution_accuracy"));
}

#endif /* GL_SGIX_convolution_accuracy */

#ifdef GL_SGIX_cube_map

static void _glewInfo_GL_SGIX_cube_map (void)
{
  glewPrintExt("GL_SGIX_cube_map", GLEW_SGIX_cube_map, glewIsSupported("GL_SGIX_cube_map"), glewGetExtension("GL_SGIX_cube_map"));
}

#endif /* GL_SGIX_cube_map */

#ifdef GL_SGIX_cylinder_texgen

static void _glewInfo_GL_SGIX_cylinder_texgen (void)
{
  glewPrintExt("GL_SGIX_cylinder_texgen", GLEW_SGIX_cylinder_texgen, glewIsSupported("GL_SGIX_cylinder_texgen"), glewGetExtension("GL_SGIX_cylinder_texgen"));
}

#endif /* GL_SGIX_cylinder_texgen */

#ifdef GL_SGIX_datapipe

static void _glewInfo_GL_SGIX_datapipe (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_datapipe", GLEW_SGIX_datapipe, glewIsSupported("GL_SGIX_datapipe"), glewGetExtension("GL_SGIX_datapipe"));

  glewInfoFunc(fi, "glAddressSpace", glAddressSpace == NULL);
  glewInfoFunc(fi, "glDataPipe", glDataPipe == NULL);
}

#endif /* GL_SGIX_datapipe */

#ifdef GL_SGIX_decimation

static void _glewInfo_GL_SGIX_decimation (void)
{
  glewPrintExt("GL_SGIX_decimation", GLEW_SGIX_decimation, glewIsSupported("GL_SGIX_decimation"), glewGetExtension("GL_SGIX_decimation"));
}

#endif /* GL_SGIX_decimation */

#ifdef GL_SGIX_depth_pass_instrument

static void _glewInfo_GL_SGIX_depth_pass_instrument (void)
{
  glewPrintExt("GL_SGIX_depth_pass_instrument", GLEW_SGIX_depth_pass_instrument, glewIsSupported("GL_SGIX_depth_pass_instrument"), glewGetExtension("GL_SGIX_depth_pass_instrument"));
}

#endif /* GL_SGIX_depth_pass_instrument */

#ifdef GL_SGIX_depth_texture

static void _glewInfo_GL_SGIX_depth_texture (void)
{
  glewPrintExt("GL_SGIX_depth_texture", GLEW_SGIX_depth_texture, glewIsSupported("GL_SGIX_depth_texture"), glewGetExtension("GL_SGIX_depth_texture"));
}

#endif /* GL_SGIX_depth_texture */

#ifdef GL_SGIX_dvc

static void _glewInfo_GL_SGIX_dvc (void)
{
  glewPrintExt("GL_SGIX_dvc", GLEW_SGIX_dvc, glewIsSupported("GL_SGIX_dvc"), glewGetExtension("GL_SGIX_dvc"));
}

#endif /* GL_SGIX_dvc */

#ifdef GL_SGIX_flush_raster

static void _glewInfo_GL_SGIX_flush_raster (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_flush_raster", GLEW_SGIX_flush_raster, glewIsSupported("GL_SGIX_flush_raster"), glewGetExtension("GL_SGIX_flush_raster"));

  glewInfoFunc(fi, "glFlushRasterSGIX", glFlushRasterSGIX == NULL);
}

#endif /* GL_SGIX_flush_raster */

#ifdef GL_SGIX_fog_blend

static void _glewInfo_GL_SGIX_fog_blend (void)
{
  glewPrintExt("GL_SGIX_fog_blend", GLEW_SGIX_fog_blend, glewIsSupported("GL_SGIX_fog_blend"), glewGetExtension("GL_SGIX_fog_blend"));
}

#endif /* GL_SGIX_fog_blend */

#ifdef GL_SGIX_fog_factor_to_alpha

static void _glewInfo_GL_SGIX_fog_factor_to_alpha (void)
{
  glewPrintExt("GL_SGIX_fog_factor_to_alpha", GLEW_SGIX_fog_factor_to_alpha, glewIsSupported("GL_SGIX_fog_factor_to_alpha"), glewGetExtension("GL_SGIX_fog_factor_to_alpha"));
}

#endif /* GL_SGIX_fog_factor_to_alpha */

#ifdef GL_SGIX_fog_layers

static void _glewInfo_GL_SGIX_fog_layers (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_fog_layers", GLEW_SGIX_fog_layers, glewIsSupported("GL_SGIX_fog_layers"), glewGetExtension("GL_SGIX_fog_layers"));

  glewInfoFunc(fi, "glFogLayersSGIX", glFogLayersSGIX == NULL);
  glewInfoFunc(fi, "glGetFogLayersSGIX", glGetFogLayersSGIX == NULL);
}

#endif /* GL_SGIX_fog_layers */

#ifdef GL_SGIX_fog_offset

static void _glewInfo_GL_SGIX_fog_offset (void)
{
  glewPrintExt("GL_SGIX_fog_offset", GLEW_SGIX_fog_offset, glewIsSupported("GL_SGIX_fog_offset"), glewGetExtension("GL_SGIX_fog_offset"));
}

#endif /* GL_SGIX_fog_offset */

#ifdef GL_SGIX_fog_patchy

static void _glewInfo_GL_SGIX_fog_patchy (void)
{
  glewPrintExt("GL_SGIX_fog_patchy", GLEW_SGIX_fog_patchy, glewIsSupported("GL_SGIX_fog_patchy"), glewGetExtension("GL_SGIX_fog_patchy"));
}

#endif /* GL_SGIX_fog_patchy */

#ifdef GL_SGIX_fog_scale

static void _glewInfo_GL_SGIX_fog_scale (void)
{
  glewPrintExt("GL_SGIX_fog_scale", GLEW_SGIX_fog_scale, glewIsSupported("GL_SGIX_fog_scale"), glewGetExtension("GL_SGIX_fog_scale"));
}

#endif /* GL_SGIX_fog_scale */

#ifdef GL_SGIX_fog_texture

static void _glewInfo_GL_SGIX_fog_texture (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_fog_texture", GLEW_SGIX_fog_texture, glewIsSupported("GL_SGIX_fog_texture"), glewGetExtension("GL_SGIX_fog_texture"));

  glewInfoFunc(fi, "glTextureFogSGIX", glTextureFogSGIX == NULL);
}

#endif /* GL_SGIX_fog_texture */

#ifdef GL_SGIX_fragment_lighting_space

static void _glewInfo_GL_SGIX_fragment_lighting_space (void)
{
  glewPrintExt("GL_SGIX_fragment_lighting_space", GLEW_SGIX_fragment_lighting_space, glewIsSupported("GL_SGIX_fragment_lighting_space"), glewGetExtension("GL_SGIX_fragment_lighting_space"));
}

#endif /* GL_SGIX_fragment_lighting_space */

#ifdef GL_SGIX_fragment_specular_lighting

static void _glewInfo_GL_SGIX_fragment_specular_lighting (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_fragment_specular_lighting", GLEW_SGIX_fragment_specular_lighting, glewIsSupported("GL_SGIX_fragment_specular_lighting"), glewGetExtension("GL_SGIX_fragment_specular_lighting"));

  glewInfoFunc(fi, "glFragmentColorMaterialSGIX", glFragmentColorMaterialSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightModelfSGIX", glFragmentLightModelfSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightModelfvSGIX", glFragmentLightModelfvSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightModeliSGIX", glFragmentLightModeliSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightModelivSGIX", glFragmentLightModelivSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightfSGIX", glFragmentLightfSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightfvSGIX", glFragmentLightfvSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightiSGIX", glFragmentLightiSGIX == NULL);
  glewInfoFunc(fi, "glFragmentLightivSGIX", glFragmentLightivSGIX == NULL);
  glewInfoFunc(fi, "glFragmentMaterialfSGIX", glFragmentMaterialfSGIX == NULL);
  glewInfoFunc(fi, "glFragmentMaterialfvSGIX", glFragmentMaterialfvSGIX == NULL);
  glewInfoFunc(fi, "glFragmentMaterialiSGIX", glFragmentMaterialiSGIX == NULL);
  glewInfoFunc(fi, "glFragmentMaterialivSGIX", glFragmentMaterialivSGIX == NULL);
  glewInfoFunc(fi, "glGetFragmentLightfvSGIX", glGetFragmentLightfvSGIX == NULL);
  glewInfoFunc(fi, "glGetFragmentLightivSGIX", glGetFragmentLightivSGIX == NULL);
  glewInfoFunc(fi, "glGetFragmentMaterialfvSGIX", glGetFragmentMaterialfvSGIX == NULL);
  glewInfoFunc(fi, "glGetFragmentMaterialivSGIX", glGetFragmentMaterialivSGIX == NULL);
}

#endif /* GL_SGIX_fragment_specular_lighting */

#ifdef GL_SGIX_fragments_instrument

static void _glewInfo_GL_SGIX_fragments_instrument (void)
{
  glewPrintExt("GL_SGIX_fragments_instrument", GLEW_SGIX_fragments_instrument, glewIsSupported("GL_SGIX_fragments_instrument"), glewGetExtension("GL_SGIX_fragments_instrument"));
}

#endif /* GL_SGIX_fragments_instrument */

#ifdef GL_SGIX_framezoom

static void _glewInfo_GL_SGIX_framezoom (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_framezoom", GLEW_SGIX_framezoom, glewIsSupported("GL_SGIX_framezoom"), glewGetExtension("GL_SGIX_framezoom"));

  glewInfoFunc(fi, "glFrameZoomSGIX", glFrameZoomSGIX == NULL);
}

#endif /* GL_SGIX_framezoom */

#ifdef GL_SGIX_icc_texture

static void _glewInfo_GL_SGIX_icc_texture (void)
{
  glewPrintExt("GL_SGIX_icc_texture", GLEW_SGIX_icc_texture, glewIsSupported("GL_SGIX_icc_texture"), glewGetExtension("GL_SGIX_icc_texture"));
}

#endif /* GL_SGIX_icc_texture */

#ifdef GL_SGIX_igloo_interface

static void _glewInfo_GL_SGIX_igloo_interface (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_igloo_interface", GLEW_SGIX_igloo_interface, glewIsSupported("GL_SGIX_igloo_interface"), glewGetExtension("GL_SGIX_igloo_interface"));

  glewInfoFunc(fi, "glIglooInterfaceSGIX", glIglooInterfaceSGIX == NULL);
}

#endif /* GL_SGIX_igloo_interface */

#ifdef GL_SGIX_image_compression

static void _glewInfo_GL_SGIX_image_compression (void)
{
  glewPrintExt("GL_SGIX_image_compression", GLEW_SGIX_image_compression, glewIsSupported("GL_SGIX_image_compression"), glewGetExtension("GL_SGIX_image_compression"));
}

#endif /* GL_SGIX_image_compression */

#ifdef GL_SGIX_impact_pixel_texture

static void _glewInfo_GL_SGIX_impact_pixel_texture (void)
{
  glewPrintExt("GL_SGIX_impact_pixel_texture", GLEW_SGIX_impact_pixel_texture, glewIsSupported("GL_SGIX_impact_pixel_texture"), glewGetExtension("GL_SGIX_impact_pixel_texture"));
}

#endif /* GL_SGIX_impact_pixel_texture */

#ifdef GL_SGIX_instrument_error

static void _glewInfo_GL_SGIX_instrument_error (void)
{
  glewPrintExt("GL_SGIX_instrument_error", GLEW_SGIX_instrument_error, glewIsSupported("GL_SGIX_instrument_error"), glewGetExtension("GL_SGIX_instrument_error"));
}

#endif /* GL_SGIX_instrument_error */

#ifdef GL_SGIX_interlace

static void _glewInfo_GL_SGIX_interlace (void)
{
  glewPrintExt("GL_SGIX_interlace", GLEW_SGIX_interlace, glewIsSupported("GL_SGIX_interlace"), glewGetExtension("GL_SGIX_interlace"));
}

#endif /* GL_SGIX_interlace */

#ifdef GL_SGIX_ir_instrument1

static void _glewInfo_GL_SGIX_ir_instrument1 (void)
{
  glewPrintExt("GL_SGIX_ir_instrument1", GLEW_SGIX_ir_instrument1, glewIsSupported("GL_SGIX_ir_instrument1"), glewGetExtension("GL_SGIX_ir_instrument1"));
}

#endif /* GL_SGIX_ir_instrument1 */

#ifdef GL_SGIX_line_quality_hint

static void _glewInfo_GL_SGIX_line_quality_hint (void)
{
  glewPrintExt("GL_SGIX_line_quality_hint", GLEW_SGIX_line_quality_hint, glewIsSupported("GL_SGIX_line_quality_hint"), glewGetExtension("GL_SGIX_line_quality_hint"));
}

#endif /* GL_SGIX_line_quality_hint */

#ifdef GL_SGIX_list_priority

static void _glewInfo_GL_SGIX_list_priority (void)
{
  glewPrintExt("GL_SGIX_list_priority", GLEW_SGIX_list_priority, glewIsSupported("GL_SGIX_list_priority"), glewGetExtension("GL_SGIX_list_priority"));
}

#endif /* GL_SGIX_list_priority */

#ifdef GL_SGIX_mpeg1

static void _glewInfo_GL_SGIX_mpeg1 (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_mpeg1", GLEW_SGIX_mpeg1, glewIsSupported("GL_SGIX_mpeg1"), glewGetExtension("GL_SGIX_mpeg1"));

  glewInfoFunc(fi, "glAllocMPEGPredictorsSGIX", glAllocMPEGPredictorsSGIX == NULL);
  glewInfoFunc(fi, "glDeleteMPEGPredictorsSGIX", glDeleteMPEGPredictorsSGIX == NULL);
  glewInfoFunc(fi, "glGenMPEGPredictorsSGIX", glGenMPEGPredictorsSGIX == NULL);
  glewInfoFunc(fi, "glGetMPEGParameterfvSGIX", glGetMPEGParameterfvSGIX == NULL);
  glewInfoFunc(fi, "glGetMPEGParameterivSGIX", glGetMPEGParameterivSGIX == NULL);
  glewInfoFunc(fi, "glGetMPEGPredictorSGIX", glGetMPEGPredictorSGIX == NULL);
  glewInfoFunc(fi, "glGetMPEGQuantTableubv", glGetMPEGQuantTableubv == NULL);
  glewInfoFunc(fi, "glIsMPEGPredictorSGIX", glIsMPEGPredictorSGIX == NULL);
  glewInfoFunc(fi, "glMPEGPredictorSGIX", glMPEGPredictorSGIX == NULL);
  glewInfoFunc(fi, "glMPEGQuantTableubv", glMPEGQuantTableubv == NULL);
  glewInfoFunc(fi, "glSwapMPEGPredictorsSGIX", glSwapMPEGPredictorsSGIX == NULL);
}

#endif /* GL_SGIX_mpeg1 */

#ifdef GL_SGIX_mpeg2

static void _glewInfo_GL_SGIX_mpeg2 (void)
{
  glewPrintExt("GL_SGIX_mpeg2", GLEW_SGIX_mpeg2, glewIsSupported("GL_SGIX_mpeg2"), glewGetExtension("GL_SGIX_mpeg2"));
}

#endif /* GL_SGIX_mpeg2 */

#ifdef GL_SGIX_nonlinear_lighting_pervertex

static void _glewInfo_GL_SGIX_nonlinear_lighting_pervertex (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_nonlinear_lighting_pervertex", GLEW_SGIX_nonlinear_lighting_pervertex, glewIsSupported("GL_SGIX_nonlinear_lighting_pervertex"), glewGetExtension("GL_SGIX_nonlinear_lighting_pervertex"));

  glewInfoFunc(fi, "glGetNonlinLightfvSGIX", glGetNonlinLightfvSGIX == NULL);
  glewInfoFunc(fi, "glGetNonlinMaterialfvSGIX", glGetNonlinMaterialfvSGIX == NULL);
  glewInfoFunc(fi, "glNonlinLightfvSGIX", glNonlinLightfvSGIX == NULL);
  glewInfoFunc(fi, "glNonlinMaterialfvSGIX", glNonlinMaterialfvSGIX == NULL);
}

#endif /* GL_SGIX_nonlinear_lighting_pervertex */

#ifdef GL_SGIX_nurbs_eval

static void _glewInfo_GL_SGIX_nurbs_eval (void)
{
  glewPrintExt("GL_SGIX_nurbs_eval", GLEW_SGIX_nurbs_eval, glewIsSupported("GL_SGIX_nurbs_eval"), glewGetExtension("GL_SGIX_nurbs_eval"));
}

#endif /* GL_SGIX_nurbs_eval */

#ifdef GL_SGIX_occlusion_instrument

static void _glewInfo_GL_SGIX_occlusion_instrument (void)
{
  glewPrintExt("GL_SGIX_occlusion_instrument", GLEW_SGIX_occlusion_instrument, glewIsSupported("GL_SGIX_occlusion_instrument"), glewGetExtension("GL_SGIX_occlusion_instrument"));
}

#endif /* GL_SGIX_occlusion_instrument */

#ifdef GL_SGIX_packed_6bytes

static void _glewInfo_GL_SGIX_packed_6bytes (void)
{
  glewPrintExt("GL_SGIX_packed_6bytes", GLEW_SGIX_packed_6bytes, glewIsSupported("GL_SGIX_packed_6bytes"), glewGetExtension("GL_SGIX_packed_6bytes"));
}

#endif /* GL_SGIX_packed_6bytes */

#ifdef GL_SGIX_pixel_texture

static void _glewInfo_GL_SGIX_pixel_texture (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_pixel_texture", GLEW_SGIX_pixel_texture, glewIsSupported("GL_SGIX_pixel_texture"), glewGetExtension("GL_SGIX_pixel_texture"));

  glewInfoFunc(fi, "glPixelTexGenSGIX", glPixelTexGenSGIX == NULL);
}

#endif /* GL_SGIX_pixel_texture */

#ifdef GL_SGIX_pixel_texture_bits

static void _glewInfo_GL_SGIX_pixel_texture_bits (void)
{
  glewPrintExt("GL_SGIX_pixel_texture_bits", GLEW_SGIX_pixel_texture_bits, glewIsSupported("GL_SGIX_pixel_texture_bits"), glewGetExtension("GL_SGIX_pixel_texture_bits"));
}

#endif /* GL_SGIX_pixel_texture_bits */

#ifdef GL_SGIX_pixel_texture_lod

static void _glewInfo_GL_SGIX_pixel_texture_lod (void)
{
  glewPrintExt("GL_SGIX_pixel_texture_lod", GLEW_SGIX_pixel_texture_lod, glewIsSupported("GL_SGIX_pixel_texture_lod"), glewGetExtension("GL_SGIX_pixel_texture_lod"));
}

#endif /* GL_SGIX_pixel_texture_lod */

#ifdef GL_SGIX_pixel_tiles

static void _glewInfo_GL_SGIX_pixel_tiles (void)
{
  glewPrintExt("GL_SGIX_pixel_tiles", GLEW_SGIX_pixel_tiles, glewIsSupported("GL_SGIX_pixel_tiles"), glewGetExtension("GL_SGIX_pixel_tiles"));
}

#endif /* GL_SGIX_pixel_tiles */

#ifdef GL_SGIX_polynomial_ffd

static void _glewInfo_GL_SGIX_polynomial_ffd (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_polynomial_ffd", GLEW_SGIX_polynomial_ffd, glewIsSupported("GL_SGIX_polynomial_ffd"), glewGetExtension("GL_SGIX_polynomial_ffd"));

  glewInfoFunc(fi, "glDeformSGIX", glDeformSGIX == NULL);
  glewInfoFunc(fi, "glLoadIdentityDeformationMapSGIX", glLoadIdentityDeformationMapSGIX == NULL);
}

#endif /* GL_SGIX_polynomial_ffd */

#ifdef GL_SGIX_quad_mesh

static void _glewInfo_GL_SGIX_quad_mesh (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_quad_mesh", GLEW_SGIX_quad_mesh, glewIsSupported("GL_SGIX_quad_mesh"), glewGetExtension("GL_SGIX_quad_mesh"));

  glewInfoFunc(fi, "glMeshBreadthSGIX", glMeshBreadthSGIX == NULL);
  glewInfoFunc(fi, "glMeshStrideSGIX", glMeshStrideSGIX == NULL);
}

#endif /* GL_SGIX_quad_mesh */

#ifdef GL_SGIX_reference_plane

static void _glewInfo_GL_SGIX_reference_plane (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_reference_plane", GLEW_SGIX_reference_plane, glewIsSupported("GL_SGIX_reference_plane"), glewGetExtension("GL_SGIX_reference_plane"));

  glewInfoFunc(fi, "glReferencePlaneSGIX", glReferencePlaneSGIX == NULL);
}

#endif /* GL_SGIX_reference_plane */

#ifdef GL_SGIX_resample

static void _glewInfo_GL_SGIX_resample (void)
{
  glewPrintExt("GL_SGIX_resample", GLEW_SGIX_resample, glewIsSupported("GL_SGIX_resample"), glewGetExtension("GL_SGIX_resample"));
}

#endif /* GL_SGIX_resample */

#ifdef GL_SGIX_scalebias_hint

static void _glewInfo_GL_SGIX_scalebias_hint (void)
{
  glewPrintExt("GL_SGIX_scalebias_hint", GLEW_SGIX_scalebias_hint, glewIsSupported("GL_SGIX_scalebias_hint"), glewGetExtension("GL_SGIX_scalebias_hint"));
}

#endif /* GL_SGIX_scalebias_hint */

#ifdef GL_SGIX_shadow

static void _glewInfo_GL_SGIX_shadow (void)
{
  glewPrintExt("GL_SGIX_shadow", GLEW_SGIX_shadow, glewIsSupported("GL_SGIX_shadow"), glewGetExtension("GL_SGIX_shadow"));
}

#endif /* GL_SGIX_shadow */

#ifdef GL_SGIX_shadow_ambient

static void _glewInfo_GL_SGIX_shadow_ambient (void)
{
  glewPrintExt("GL_SGIX_shadow_ambient", GLEW_SGIX_shadow_ambient, glewIsSupported("GL_SGIX_shadow_ambient"), glewGetExtension("GL_SGIX_shadow_ambient"));
}

#endif /* GL_SGIX_shadow_ambient */

#ifdef GL_SGIX_slim

static void _glewInfo_GL_SGIX_slim (void)
{
  glewPrintExt("GL_SGIX_slim", GLEW_SGIX_slim, glewIsSupported("GL_SGIX_slim"), glewGetExtension("GL_SGIX_slim"));
}

#endif /* GL_SGIX_slim */

#ifdef GL_SGIX_spotlight_cutoff

static void _glewInfo_GL_SGIX_spotlight_cutoff (void)
{
  glewPrintExt("GL_SGIX_spotlight_cutoff", GLEW_SGIX_spotlight_cutoff, glewIsSupported("GL_SGIX_spotlight_cutoff"), glewGetExtension("GL_SGIX_spotlight_cutoff"));
}

#endif /* GL_SGIX_spotlight_cutoff */

#ifdef GL_SGIX_sprite

static void _glewInfo_GL_SGIX_sprite (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_sprite", GLEW_SGIX_sprite, glewIsSupported("GL_SGIX_sprite"), glewGetExtension("GL_SGIX_sprite"));

  glewInfoFunc(fi, "glSpriteParameterfSGIX", glSpriteParameterfSGIX == NULL);
  glewInfoFunc(fi, "glSpriteParameterfvSGIX", glSpriteParameterfvSGIX == NULL);
  glewInfoFunc(fi, "glSpriteParameteriSGIX", glSpriteParameteriSGIX == NULL);
  glewInfoFunc(fi, "glSpriteParameterivSGIX", glSpriteParameterivSGIX == NULL);
}

#endif /* GL_SGIX_sprite */

#ifdef GL_SGIX_subdiv_patch

static void _glewInfo_GL_SGIX_subdiv_patch (void)
{
  glewPrintExt("GL_SGIX_subdiv_patch", GLEW_SGIX_subdiv_patch, glewIsSupported("GL_SGIX_subdiv_patch"), glewGetExtension("GL_SGIX_subdiv_patch"));
}

#endif /* GL_SGIX_subdiv_patch */

#ifdef GL_SGIX_subsample

static void _glewInfo_GL_SGIX_subsample (void)
{
  glewPrintExt("GL_SGIX_subsample", GLEW_SGIX_subsample, glewIsSupported("GL_SGIX_subsample"), glewGetExtension("GL_SGIX_subsample"));
}

#endif /* GL_SGIX_subsample */

#ifdef GL_SGIX_tag_sample_buffer

static void _glewInfo_GL_SGIX_tag_sample_buffer (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_tag_sample_buffer", GLEW_SGIX_tag_sample_buffer, glewIsSupported("GL_SGIX_tag_sample_buffer"), glewGetExtension("GL_SGIX_tag_sample_buffer"));

  glewInfoFunc(fi, "glTagSampleBufferSGIX", glTagSampleBufferSGIX == NULL);
}

#endif /* GL_SGIX_tag_sample_buffer */

#ifdef GL_SGIX_texture_add_env

static void _glewInfo_GL_SGIX_texture_add_env (void)
{
  glewPrintExt("GL_SGIX_texture_add_env", GLEW_SGIX_texture_add_env, glewIsSupported("GL_SGIX_texture_add_env"), glewGetExtension("GL_SGIX_texture_add_env"));
}

#endif /* GL_SGIX_texture_add_env */

#ifdef GL_SGIX_texture_coordinate_clamp

static void _glewInfo_GL_SGIX_texture_coordinate_clamp (void)
{
  glewPrintExt("GL_SGIX_texture_coordinate_clamp", GLEW_SGIX_texture_coordinate_clamp, glewIsSupported("GL_SGIX_texture_coordinate_clamp"), glewGetExtension("GL_SGIX_texture_coordinate_clamp"));
}

#endif /* GL_SGIX_texture_coordinate_clamp */

#ifdef GL_SGIX_texture_lod_bias

static void _glewInfo_GL_SGIX_texture_lod_bias (void)
{
  glewPrintExt("GL_SGIX_texture_lod_bias", GLEW_SGIX_texture_lod_bias, glewIsSupported("GL_SGIX_texture_lod_bias"), glewGetExtension("GL_SGIX_texture_lod_bias"));
}

#endif /* GL_SGIX_texture_lod_bias */

#ifdef GL_SGIX_texture_mipmap_anisotropic

static void _glewInfo_GL_SGIX_texture_mipmap_anisotropic (void)
{
  glewPrintExt("GL_SGIX_texture_mipmap_anisotropic", GLEW_SGIX_texture_mipmap_anisotropic, glewIsSupported("GL_SGIX_texture_mipmap_anisotropic"), glewGetExtension("GL_SGIX_texture_mipmap_anisotropic"));
}

#endif /* GL_SGIX_texture_mipmap_anisotropic */

#ifdef GL_SGIX_texture_multi_buffer

static void _glewInfo_GL_SGIX_texture_multi_buffer (void)
{
  glewPrintExt("GL_SGIX_texture_multi_buffer", GLEW_SGIX_texture_multi_buffer, glewIsSupported("GL_SGIX_texture_multi_buffer"), glewGetExtension("GL_SGIX_texture_multi_buffer"));
}

#endif /* GL_SGIX_texture_multi_buffer */

#ifdef GL_SGIX_texture_phase

static void _glewInfo_GL_SGIX_texture_phase (void)
{
  glewPrintExt("GL_SGIX_texture_phase", GLEW_SGIX_texture_phase, glewIsSupported("GL_SGIX_texture_phase"), glewGetExtension("GL_SGIX_texture_phase"));
}

#endif /* GL_SGIX_texture_phase */

#ifdef GL_SGIX_texture_range

static void _glewInfo_GL_SGIX_texture_range (void)
{
  glewPrintExt("GL_SGIX_texture_range", GLEW_SGIX_texture_range, glewIsSupported("GL_SGIX_texture_range"), glewGetExtension("GL_SGIX_texture_range"));
}

#endif /* GL_SGIX_texture_range */

#ifdef GL_SGIX_texture_scale_bias

static void _glewInfo_GL_SGIX_texture_scale_bias (void)
{
  glewPrintExt("GL_SGIX_texture_scale_bias", GLEW_SGIX_texture_scale_bias, glewIsSupported("GL_SGIX_texture_scale_bias"), glewGetExtension("GL_SGIX_texture_scale_bias"));
}

#endif /* GL_SGIX_texture_scale_bias */

#ifdef GL_SGIX_texture_supersample

static void _glewInfo_GL_SGIX_texture_supersample (void)
{
  glewPrintExt("GL_SGIX_texture_supersample", GLEW_SGIX_texture_supersample, glewIsSupported("GL_SGIX_texture_supersample"), glewGetExtension("GL_SGIX_texture_supersample"));
}

#endif /* GL_SGIX_texture_supersample */

#ifdef GL_SGIX_vector_ops

static void _glewInfo_GL_SGIX_vector_ops (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_vector_ops", GLEW_SGIX_vector_ops, glewIsSupported("GL_SGIX_vector_ops"), glewGetExtension("GL_SGIX_vector_ops"));

  glewInfoFunc(fi, "glGetVectorOperationSGIX", glGetVectorOperationSGIX == NULL);
  glewInfoFunc(fi, "glVectorOperationSGIX", glVectorOperationSGIX == NULL);
}

#endif /* GL_SGIX_vector_ops */

#ifdef GL_SGIX_vertex_array_object

static void _glewInfo_GL_SGIX_vertex_array_object (void)
{
  GLboolean fi = glewPrintExt("GL_SGIX_vertex_array_object", GLEW_SGIX_vertex_array_object, glewIsSupported("GL_SGIX_vertex_array_object"), glewGetExtension("GL_SGIX_vertex_array_object"));

  glewInfoFunc(fi, "glAreVertexArraysResidentSGIX", glAreVertexArraysResidentSGIX == NULL);
  glewInfoFunc(fi, "glBindVertexArraySGIX", glBindVertexArraySGIX == NULL);
  glewInfoFunc(fi, "glDeleteVertexArraysSGIX", glDeleteVertexArraysSGIX == NULL);
  glewInfoFunc(fi, "glGenVertexArraysSGIX", glGenVertexArraysSGIX == NULL);
  glewInfoFunc(fi, "glIsVertexArraySGIX", glIsVertexArraySGIX == NULL);
  glewInfoFunc(fi, "glPrioritizeVertexArraysSGIX", glPrioritizeVertexArraysSGIX == NULL);
}

#endif /* GL_SGIX_vertex_array_object */

#ifdef GL_SGIX_vertex_preclip

static void _glewInfo_GL_SGIX_vertex_preclip (void)
{
  glewPrintExt("GL_SGIX_vertex_preclip", GLEW_SGIX_vertex_preclip, glewIsSupported("GL_SGIX_vertex_preclip"), glewGetExtension("GL_SGIX_vertex_preclip"));
}

#endif /* GL_SGIX_vertex_preclip */

#ifdef GL_SGIX_vertex_preclip_hint

static void _glewInfo_GL_SGIX_vertex_preclip_hint (void)
{
  glewPrintExt("GL_SGIX_vertex_preclip_hint", GLEW_SGIX_vertex_preclip_hint, glewIsSupported("GL_SGIX_vertex_preclip_hint"), glewGetExtension("GL_SGIX_vertex_preclip_hint"));
}

#endif /* GL_SGIX_vertex_preclip_hint */

#ifdef GL_SGIX_ycrcb

static void _glewInfo_GL_SGIX_ycrcb (void)
{
  glewPrintExt("GL_SGIX_ycrcb", GLEW_SGIX_ycrcb, glewIsSupported("GL_SGIX_ycrcb"), glewGetExtension("GL_SGIX_ycrcb"));
}

#endif /* GL_SGIX_ycrcb */

#ifdef GL_SGIX_ycrcb_subsample

static void _glewInfo_GL_SGIX_ycrcb_subsample (void)
{
  glewPrintExt("GL_SGIX_ycrcb_subsample", GLEW_SGIX_ycrcb_subsample, glewIsSupported("GL_SGIX_ycrcb_subsample"), glewGetExtension("GL_SGIX_ycrcb_subsample"));
}

#endif /* GL_SGIX_ycrcb_subsample */

#ifdef GL_SGIX_ycrcba

static void _glewInfo_GL_SGIX_ycrcba (void)
{
  glewPrintExt("GL_SGIX_ycrcba", GLEW_SGIX_ycrcba, glewIsSupported("GL_SGIX_ycrcba"), glewGetExtension("GL_SGIX_ycrcba"));
}

#endif /* GL_SGIX_ycrcba */

#ifdef GL_SGI_color_matrix

static void _glewInfo_GL_SGI_color_matrix (void)
{
  glewPrintExt("GL_SGI_color_matrix", GLEW_SGI_color_matrix, glewIsSupported("GL_SGI_color_matrix"), glewGetExtension("GL_SGI_color_matrix"));
}

#endif /* GL_SGI_color_matrix */

#ifdef GL_SGI_color_table

static void _glewInfo_GL_SGI_color_table (void)
{
  GLboolean fi = glewPrintExt("GL_SGI_color_table", GLEW_SGI_color_table, glewIsSupported("GL_SGI_color_table"), glewGetExtension("GL_SGI_color_table"));

  glewInfoFunc(fi, "glColorTableParameterfvSGI", glColorTableParameterfvSGI == NULL);
  glewInfoFunc(fi, "glColorTableParameterivSGI", glColorTableParameterivSGI == NULL);
  glewInfoFunc(fi, "glColorTableSGI", glColorTableSGI == NULL);
  glewInfoFunc(fi, "glCopyColorTableSGI", glCopyColorTableSGI == NULL);
  glewInfoFunc(fi, "glGetColorTableParameterfvSGI", glGetColorTableParameterfvSGI == NULL);
  glewInfoFunc(fi, "glGetColorTableParameterivSGI", glGetColorTableParameterivSGI == NULL);
  glewInfoFunc(fi, "glGetColorTableSGI", glGetColorTableSGI == NULL);
}

#endif /* GL_SGI_color_table */

#ifdef GL_SGI_complex

static void _glewInfo_GL_SGI_complex (void)
{
  glewPrintExt("GL_SGI_complex", GLEW_SGI_complex, glewIsSupported("GL_SGI_complex"), glewGetExtension("GL_SGI_complex"));
}

#endif /* GL_SGI_complex */

#ifdef GL_SGI_complex_type

static void _glewInfo_GL_SGI_complex_type (void)
{
  glewPrintExt("GL_SGI_complex_type", GLEW_SGI_complex_type, glewIsSupported("GL_SGI_complex_type"), glewGetExtension("GL_SGI_complex_type"));
}

#endif /* GL_SGI_complex_type */

#ifdef GL_SGI_fft

static void _glewInfo_GL_SGI_fft (void)
{
  GLboolean fi = glewPrintExt("GL_SGI_fft", GLEW_SGI_fft, glewIsSupported("GL_SGI_fft"), glewGetExtension("GL_SGI_fft"));

  glewInfoFunc(fi, "glGetPixelTransformParameterfvSGI", glGetPixelTransformParameterfvSGI == NULL);
  glewInfoFunc(fi, "glGetPixelTransformParameterivSGI", glGetPixelTransformParameterivSGI == NULL);
  glewInfoFunc(fi, "glPixelTransformParameterfSGI", glPixelTransformParameterfSGI == NULL);
  glewInfoFunc(fi, "glPixelTransformParameterfvSGI", glPixelTransformParameterfvSGI == NULL);
  glewInfoFunc(fi, "glPixelTransformParameteriSGI", glPixelTransformParameteriSGI == NULL);
  glewInfoFunc(fi, "glPixelTransformParameterivSGI", glPixelTransformParameterivSGI == NULL);
  glewInfoFunc(fi, "glPixelTransformSGI", glPixelTransformSGI == NULL);
}

#endif /* GL_SGI_fft */

#ifdef GL_SGI_texture_color_table

static void _glewInfo_GL_SGI_texture_color_table (void)
{
  glewPrintExt("GL_SGI_texture_color_table", GLEW_SGI_texture_color_table, glewIsSupported("GL_SGI_texture_color_table"), glewGetExtension("GL_SGI_texture_color_table"));
}

#endif /* GL_SGI_texture_color_table */

#ifdef GL_SUNX_constant_data

static void _glewInfo_GL_SUNX_constant_data (void)
{
  GLboolean fi = glewPrintExt("GL_SUNX_constant_data", GLEW_SUNX_constant_data, glewIsSupported("GL_SUNX_constant_data"), glewGetExtension("GL_SUNX_constant_data"));

  glewInfoFunc(fi, "glFinishTextureSUNX", glFinishTextureSUNX == NULL);
}

#endif /* GL_SUNX_constant_data */

#ifdef GL_SUN_convolution_border_modes

static void _glewInfo_GL_SUN_convolution_border_modes (void)
{
  glewPrintExt("GL_SUN_convolution_border_modes", GLEW_SUN_convolution_border_modes, glewIsSupported("GL_SUN_convolution_border_modes"), glewGetExtension("GL_SUN_convolution_border_modes"));
}

#endif /* GL_SUN_convolution_border_modes */

#ifdef GL_SUN_global_alpha

static void _glewInfo_GL_SUN_global_alpha (void)
{
  GLboolean fi = glewPrintExt("GL_SUN_global_alpha", GLEW_SUN_global_alpha, glewIsSupported("GL_SUN_global_alpha"), glewGetExtension("GL_SUN_global_alpha"));

  glewInfoFunc(fi, "glGlobalAlphaFactorbSUN", glGlobalAlphaFactorbSUN == NULL);
  glewInfoFunc(fi, "glGlobalAlphaFactordSUN", glGlobalAlphaFactordSUN == NULL);
  glewInfoFunc(fi, "glGlobalAlphaFactorfSUN", glGlobalAlphaFactorfSUN == NULL);
  glewInfoFunc(fi, "glGlobalAlphaFactoriSUN", glGlobalAlphaFactoriSUN == NULL);
  glewInfoFunc(fi, "glGlobalAlphaFactorsSUN", glGlobalAlphaFactorsSUN == NULL);
  glewInfoFunc(fi, "glGlobalAlphaFactorubSUN", glGlobalAlphaFactorubSUN == NULL);
  glewInfoFunc(fi, "glGlobalAlphaFactoruiSUN", glGlobalAlphaFactoruiSUN == NULL);
  glewInfoFunc(fi, "glGlobalAlphaFactorusSUN", glGlobalAlphaFactorusSUN == NULL);
}

#endif /* GL_SUN_global_alpha */

#ifdef GL_SUN_mesh_array

static void _glewInfo_GL_SUN_mesh_array (void)
{
  glewPrintExt("GL_SUN_mesh_array", GLEW_SUN_mesh_array, glewIsSupported("GL_SUN_mesh_array"), glewGetExtension("GL_SUN_mesh_array"));
}

#endif /* GL_SUN_mesh_array */

#ifdef GL_SUN_read_video_pixels

static void _glewInfo_GL_SUN_read_video_pixels (void)
{
  GLboolean fi = glewPrintExt("GL_SUN_read_video_pixels", GLEW_SUN_read_video_pixels, glewIsSupported("GL_SUN_read_video_pixels"), glewGetExtension("GL_SUN_read_video_pixels"));

  glewInfoFunc(fi, "glReadVideoPixelsSUN", glReadVideoPixelsSUN == NULL);
}

#endif /* GL_SUN_read_video_pixels */

#ifdef GL_SUN_slice_accum

static void _glewInfo_GL_SUN_slice_accum (void)
{
  glewPrintExt("GL_SUN_slice_accum", GLEW_SUN_slice_accum, glewIsSupported("GL_SUN_slice_accum"), glewGetExtension("GL_SUN_slice_accum"));
}

#endif /* GL_SUN_slice_accum */

#ifdef GL_SUN_triangle_list

static void _glewInfo_GL_SUN_triangle_list (void)
{
  GLboolean fi = glewPrintExt("GL_SUN_triangle_list", GLEW_SUN_triangle_list, glewIsSupported("GL_SUN_triangle_list"), glewGetExtension("GL_SUN_triangle_list"));

  glewInfoFunc(fi, "glReplacementCodePointerSUN", glReplacementCodePointerSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeubSUN", glReplacementCodeubSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeubvSUN", glReplacementCodeubvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiSUN", glReplacementCodeuiSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuivSUN", glReplacementCodeuivSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeusSUN", glReplacementCodeusSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeusvSUN", glReplacementCodeusvSUN == NULL);
}

#endif /* GL_SUN_triangle_list */

#ifdef GL_SUN_vertex

static void _glewInfo_GL_SUN_vertex (void)
{
  GLboolean fi = glewPrintExt("GL_SUN_vertex", GLEW_SUN_vertex, glewIsSupported("GL_SUN_vertex"), glewGetExtension("GL_SUN_vertex"));

  glewInfoFunc(fi, "glColor3fVertex3fSUN", glColor3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glColor3fVertex3fvSUN", glColor3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glColor4fNormal3fVertex3fSUN", glColor4fNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glColor4fNormal3fVertex3fvSUN", glColor4fNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glColor4ubVertex2fSUN", glColor4ubVertex2fSUN == NULL);
  glewInfoFunc(fi, "glColor4ubVertex2fvSUN", glColor4ubVertex2fvSUN == NULL);
  glewInfoFunc(fi, "glColor4ubVertex3fSUN", glColor4ubVertex3fSUN == NULL);
  glewInfoFunc(fi, "glColor4ubVertex3fvSUN", glColor4ubVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glNormal3fVertex3fSUN", glNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glNormal3fVertex3fvSUN", glNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiColor3fVertex3fSUN", glReplacementCodeuiColor3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiColor3fVertex3fvSUN", glReplacementCodeuiColor3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiColor4fNormal3fVertex3fSUN", glReplacementCodeuiColor4fNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiColor4fNormal3fVertex3fvSUN", glReplacementCodeuiColor4fNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiColor4ubVertex3fSUN", glReplacementCodeuiColor4ubVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiColor4ubVertex3fvSUN", glReplacementCodeuiColor4ubVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiNormal3fVertex3fSUN", glReplacementCodeuiNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiNormal3fVertex3fvSUN", glReplacementCodeuiNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN", glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN", glReplacementCodeuiTexCoord2fColor4fNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiTexCoord2fNormal3fVertex3fSUN", glReplacementCodeuiTexCoord2fNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN", glReplacementCodeuiTexCoord2fNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiTexCoord2fVertex3fSUN", glReplacementCodeuiTexCoord2fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiTexCoord2fVertex3fvSUN", glReplacementCodeuiTexCoord2fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiVertex3fSUN", glReplacementCodeuiVertex3fSUN == NULL);
  glewInfoFunc(fi, "glReplacementCodeuiVertex3fvSUN", glReplacementCodeuiVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fColor3fVertex3fSUN", glTexCoord2fColor3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fColor3fVertex3fvSUN", glTexCoord2fColor3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fColor4fNormal3fVertex3fSUN", glTexCoord2fColor4fNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fColor4fNormal3fVertex3fvSUN", glTexCoord2fColor4fNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fColor4ubVertex3fSUN", glTexCoord2fColor4ubVertex3fSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fColor4ubVertex3fvSUN", glTexCoord2fColor4ubVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fNormal3fVertex3fSUN", glTexCoord2fNormal3fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fNormal3fVertex3fvSUN", glTexCoord2fNormal3fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fVertex3fSUN", glTexCoord2fVertex3fSUN == NULL);
  glewInfoFunc(fi, "glTexCoord2fVertex3fvSUN", glTexCoord2fVertex3fvSUN == NULL);
  glewInfoFunc(fi, "glTexCoord4fColor4fNormal3fVertex4fSUN", glTexCoord4fColor4fNormal3fVertex4fSUN == NULL);
  glewInfoFunc(fi, "glTexCoord4fColor4fNormal3fVertex4fvSUN", glTexCoord4fColor4fNormal3fVertex4fvSUN == NULL);
  glewInfoFunc(fi, "glTexCoord4fVertex4fSUN", glTexCoord4fVertex4fSUN == NULL);
  glewInfoFunc(fi, "glTexCoord4fVertex4fvSUN", glTexCoord4fVertex4fvSUN == NULL);
}

#endif /* GL_SUN_vertex */

#ifdef GL_VIV_shader_binary

static void _glewInfo_GL_VIV_shader_binary (void)
{
  glewPrintExt("GL_VIV_shader_binary", GLEW_VIV_shader_binary, glewIsSupported("GL_VIV_shader_binary"), glewGetExtension("GL_VIV_shader_binary"));
}

#endif /* GL_VIV_shader_binary */

#ifdef GL_WIN_phong_shading

static void _glewInfo_GL_WIN_phong_shading (void)
{
  glewPrintExt("GL_WIN_phong_shading", GLEW_WIN_phong_shading, glewIsSupported("GL_WIN_phong_shading"), glewGetExtension("GL_WIN_phong_shading"));
}

#endif /* GL_WIN_phong_shading */

#ifdef GL_WIN_scene_markerXXX

static void _glewInfo_GL_WIN_scene_markerXXX (void)
{
  glewPrintExt("GL_WIN_scene_markerXXX", GLEW_WIN_scene_markerXXX, glewIsSupported("GL_WIN_scene_markerXXX"), glewGetExtension("GL_WIN_scene_markerXXX"));
}

#endif /* GL_WIN_scene_markerXXX */

#ifdef GL_WIN_specular_fog

static void _glewInfo_GL_WIN_specular_fog (void)
{
  glewPrintExt("GL_WIN_specular_fog", GLEW_WIN_specular_fog, glewIsSupported("GL_WIN_specular_fog"), glewGetExtension("GL_WIN_specular_fog"));
}

#endif /* GL_WIN_specular_fog */

#ifdef GL_WIN_swap_hint

static void _glewInfo_GL_WIN_swap_hint (void)
{
  GLboolean fi = glewPrintExt("GL_WIN_swap_hint", GLEW_WIN_swap_hint, glewIsSupported("GL_WIN_swap_hint"), glewGetExtension("GL_WIN_swap_hint"));

  glewInfoFunc(fi, "glAddSwapHintRectWIN", glAddSwapHintRectWIN == NULL);
}

#endif /* GL_WIN_swap_hint */

#if defined(GLEW_EGL)

#ifdef EGL_VERSION_1_0

static void _glewInfo_EGL_VERSION_1_0 (void)
{
  GLboolean fi = glewPrintExt("EGL_VERSION_1_0", EGLEW_VERSION_1_0, EGLEW_VERSION_1_0, EGLEW_VERSION_1_0);

  glewInfoFunc(fi, "eglChooseConfig", eglChooseConfig == NULL);
  glewInfoFunc(fi, "eglCopyBuffers", eglCopyBuffers == NULL);
  glewInfoFunc(fi, "eglCreateContext", eglCreateContext == NULL);
  glewInfoFunc(fi, "eglCreatePbufferSurface", eglCreatePbufferSurface == NULL);
  glewInfoFunc(fi, "eglCreatePixmapSurface", eglCreatePixmapSurface == NULL);
  glewInfoFunc(fi, "eglCreateWindowSurface", eglCreateWindowSurface == NULL);
  glewInfoFunc(fi, "eglDestroyContext", eglDestroyContext == NULL);
  glewInfoFunc(fi, "eglDestroySurface", eglDestroySurface == NULL);
  glewInfoFunc(fi, "eglGetConfigAttrib", eglGetConfigAttrib == NULL);
  glewInfoFunc(fi, "eglGetConfigs", eglGetConfigs == NULL);
  glewInfoFunc(fi, "eglGetCurrentDisplay", eglGetCurrentDisplay == NULL);
  glewInfoFunc(fi, "eglGetCurrentSurface", eglGetCurrentSurface == NULL);
  glewInfoFunc(fi, "eglGetDisplay", eglGetDisplay == NULL);
  glewInfoFunc(fi, "eglGetError", eglGetError == NULL);
  glewInfoFunc(fi, "eglInitialize", eglInitialize == NULL);
  glewInfoFunc(fi, "eglMakeCurrent", eglMakeCurrent == NULL);
  glewInfoFunc(fi, "eglQueryContext", eglQueryContext == NULL);
  glewInfoFunc(fi, "eglQueryString", eglQueryString == NULL);
  glewInfoFunc(fi, "eglQuerySurface", eglQuerySurface == NULL);
  glewInfoFunc(fi, "eglSwapBuffers", eglSwapBuffers == NULL);
  glewInfoFunc(fi, "eglTerminate", eglTerminate == NULL);
  glewInfoFunc(fi, "eglWaitGL", eglWaitGL == NULL);
  glewInfoFunc(fi, "eglWaitNative", eglWaitNative == NULL);
}

#endif /* EGL_VERSION_1_0 */

#ifdef EGL_VERSION_1_1

static void _glewInfo_EGL_VERSION_1_1 (void)
{
  GLboolean fi = glewPrintExt("EGL_VERSION_1_1", EGLEW_VERSION_1_1, EGLEW_VERSION_1_1, EGLEW_VERSION_1_1);

  glewInfoFunc(fi, "eglBindTexImage", eglBindTexImage == NULL);
  glewInfoFunc(fi, "eglReleaseTexImage", eglReleaseTexImage == NULL);
  glewInfoFunc(fi, "eglSurfaceAttrib", eglSurfaceAttrib == NULL);
  glewInfoFunc(fi, "eglSwapInterval", eglSwapInterval == NULL);
}

#endif /* EGL_VERSION_1_1 */

#ifdef EGL_VERSION_1_2

static void _glewInfo_EGL_VERSION_1_2 (void)
{
  GLboolean fi = glewPrintExt("EGL_VERSION_1_2", EGLEW_VERSION_1_2, EGLEW_VERSION_1_2, EGLEW_VERSION_1_2);

  glewInfoFunc(fi, "eglBindAPI", eglBindAPI == NULL);
  glewInfoFunc(fi, "eglCreatePbufferFromClientBuffer", eglCreatePbufferFromClientBuffer == NULL);
  glewInfoFunc(fi, "eglQueryAPI", eglQueryAPI == NULL);
  glewInfoFunc(fi, "eglReleaseThread", eglReleaseThread == NULL);
  glewInfoFunc(fi, "eglWaitClient", eglWaitClient == NULL);
}

#endif /* EGL_VERSION_1_2 */

#ifdef EGL_VERSION_1_3

static void _glewInfo_EGL_VERSION_1_3 (void)
{
  glewPrintExt("EGL_VERSION_1_3", EGLEW_VERSION_1_3, EGLEW_VERSION_1_3, EGLEW_VERSION_1_3);
}

#endif /* EGL_VERSION_1_3 */

#ifdef EGL_VERSION_1_4

static void _glewInfo_EGL_VERSION_1_4 (void)
{
  GLboolean fi = glewPrintExt("EGL_VERSION_1_4", EGLEW_VERSION_1_4, EGLEW_VERSION_1_4, EGLEW_VERSION_1_4);

  glewInfoFunc(fi, "eglGetCurrentContext", eglGetCurrentContext == NULL);
}

#endif /* EGL_VERSION_1_4 */

#ifdef EGL_VERSION_1_5

static void _glewInfo_EGL_VERSION_1_5 (void)
{
  GLboolean fi = glewPrintExt("EGL_VERSION_1_5", EGLEW_VERSION_1_5, EGLEW_VERSION_1_5, EGLEW_VERSION_1_5);

  glewInfoFunc(fi, "eglClientWaitSync", eglClientWaitSync == NULL);
  glewInfoFunc(fi, "eglCreateImage", eglCreateImage == NULL);
  glewInfoFunc(fi, "eglCreatePlatformPixmapSurface", eglCreatePlatformPixmapSurface == NULL);
  glewInfoFunc(fi, "eglCreatePlatformWindowSurface", eglCreatePlatformWindowSurface == NULL);
  glewInfoFunc(fi, "eglCreateSync", eglCreateSync == NULL);
  glewInfoFunc(fi, "eglDestroyImage", eglDestroyImage == NULL);
  glewInfoFunc(fi, "eglDestroySync", eglDestroySync == NULL);
  glewInfoFunc(fi, "eglGetPlatformDisplay", eglGetPlatformDisplay == NULL);
  glewInfoFunc(fi, "eglGetSyncAttrib", eglGetSyncAttrib == NULL);
  glewInfoFunc(fi, "eglWaitSync", eglWaitSync == NULL);
}

#endif /* EGL_VERSION_1_5 */

#ifdef EGL_ANDROID_GLES_layers

static void _glewInfo_EGL_ANDROID_GLES_layers (void)
{
  glewPrintExt("EGL_ANDROID_GLES_layers", EGLEW_ANDROID_GLES_layers, eglewIsSupported("EGL_ANDROID_GLES_layers"), eglewGetExtension("EGL_ANDROID_GLES_layers"));
}

#endif /* EGL_ANDROID_GLES_layers */

#ifdef EGL_ANDROID_blob_cache

static void _glewInfo_EGL_ANDROID_blob_cache (void)
{
  GLboolean fi = glewPrintExt("EGL_ANDROID_blob_cache", EGLEW_ANDROID_blob_cache, eglewIsSupported("EGL_ANDROID_blob_cache"), eglewGetExtension("EGL_ANDROID_blob_cache"));

  glewInfoFunc(fi, "eglSetBlobCacheFuncsANDROID", eglSetBlobCacheFuncsANDROID == NULL);
}

#endif /* EGL_ANDROID_blob_cache */

#ifdef EGL_ANDROID_create_native_client_buffer

static void _glewInfo_EGL_ANDROID_create_native_client_buffer (void)
{
  GLboolean fi = glewPrintExt("EGL_ANDROID_create_native_client_buffer", EGLEW_ANDROID_create_native_client_buffer, eglewIsSupported("EGL_ANDROID_create_native_client_buffer"), eglewGetExtension("EGL_ANDROID_create_native_client_buffer"));

  glewInfoFunc(fi, "eglCreateNativeClientBufferANDROID", eglCreateNativeClientBufferANDROID == NULL);
}

#endif /* EGL_ANDROID_create_native_client_buffer */

#ifdef EGL_ANDROID_framebuffer_target

static void _glewInfo_EGL_ANDROID_framebuffer_target (void)
{
  glewPrintExt("EGL_ANDROID_framebuffer_target", EGLEW_ANDROID_framebuffer_target, eglewIsSupported("EGL_ANDROID_framebuffer_target"), eglewGetExtension("EGL_ANDROID_framebuffer_target"));
}

#endif /* EGL_ANDROID_framebuffer_target */

#ifdef EGL_ANDROID_front_buffer_auto_refresh

static void _glewInfo_EGL_ANDROID_front_buffer_auto_refresh (void)
{
  glewPrintExt("EGL_ANDROID_front_buffer_auto_refresh", EGLEW_ANDROID_front_buffer_auto_refresh, eglewIsSupported("EGL_ANDROID_front_buffer_auto_refresh"), eglewGetExtension("EGL_ANDROID_front_buffer_auto_refresh"));
}

#endif /* EGL_ANDROID_front_buffer_auto_refresh */

#ifdef EGL_ANDROID_get_frame_timestamps

static void _glewInfo_EGL_ANDROID_get_frame_timestamps (void)
{
  GLboolean fi = glewPrintExt("EGL_ANDROID_get_frame_timestamps", EGLEW_ANDROID_get_frame_timestamps, eglewIsSupported("EGL_ANDROID_get_frame_timestamps"), eglewGetExtension("EGL_ANDROID_get_frame_timestamps"));

  glewInfoFunc(fi, "eglGetCompositorTimingANDROID", eglGetCompositorTimingANDROID == NULL);
  glewInfoFunc(fi, "eglGetCompositorTimingSupportedANDROID", eglGetCompositorTimingSupportedANDROID == NULL);
  glewInfoFunc(fi, "eglGetFrameTimestampSupportedANDROID", eglGetFrameTimestampSupportedANDROID == NULL);
  glewInfoFunc(fi, "eglGetFrameTimestampsANDROID", eglGetFrameTimestampsANDROID == NULL);
  glewInfoFunc(fi, "eglGetNextFrameIdANDROID", eglGetNextFrameIdANDROID == NULL);
}

#endif /* EGL_ANDROID_get_frame_timestamps */

#ifdef EGL_ANDROID_get_native_client_buffer

static void _glewInfo_EGL_ANDROID_get_native_client_buffer (void)
{
  GLboolean fi = glewPrintExt("EGL_ANDROID_get_native_client_buffer", EGLEW_ANDROID_get_native_client_buffer, eglewIsSupported("EGL_ANDROID_get_native_client_buffer"), eglewGetExtension("EGL_ANDROID_get_native_client_buffer"));

  glewInfoFunc(fi, "eglGetNativeClientBufferANDROID", eglGetNativeClientBufferANDROID == NULL);
}

#endif /* EGL_ANDROID_get_native_client_buffer */

#ifdef EGL_ANDROID_image_native_buffer

static void _glewInfo_EGL_ANDROID_image_native_buffer (void)
{
  glewPrintExt("EGL_ANDROID_image_native_buffer", EGLEW_ANDROID_image_native_buffer, eglewIsSupported("EGL_ANDROID_image_native_buffer"), eglewGetExtension("EGL_ANDROID_image_native_buffer"));
}

#endif /* EGL_ANDROID_image_native_buffer */

#ifdef EGL_ANDROID_native_fence_sync

static void _glewInfo_EGL_ANDROID_native_fence_sync (void)
{
  GLboolean fi = glewPrintExt("EGL_ANDROID_native_fence_sync", EGLEW_ANDROID_native_fence_sync, eglewIsSupported("EGL_ANDROID_native_fence_sync"), eglewGetExtension("EGL_ANDROID_native_fence_sync"));

  glewInfoFunc(fi, "eglDupNativeFenceFDANDROID", eglDupNativeFenceFDANDROID == NULL);
}

#endif /* EGL_ANDROID_native_fence_sync */

#ifdef EGL_ANDROID_presentation_time

static void _glewInfo_EGL_ANDROID_presentation_time (void)
{
  GLboolean fi = glewPrintExt("EGL_ANDROID_presentation_time", EGLEW_ANDROID_presentation_time, eglewIsSupported("EGL_ANDROID_presentation_time"), eglewGetExtension("EGL_ANDROID_presentation_time"));

  glewInfoFunc(fi, "eglPresentationTimeANDROID", eglPresentationTimeANDROID == NULL);
}

#endif /* EGL_ANDROID_presentation_time */

#ifdef EGL_ANDROID_recordable

static void _glewInfo_EGL_ANDROID_recordable (void)
{
  glewPrintExt("EGL_ANDROID_recordable", EGLEW_ANDROID_recordable, eglewIsSupported("EGL_ANDROID_recordable"), eglewGetExtension("EGL_ANDROID_recordable"));
}

#endif /* EGL_ANDROID_recordable */

#ifdef EGL_ANGLE_d3d_share_handle_client_buffer

static void _glewInfo_EGL_ANGLE_d3d_share_handle_client_buffer (void)
{
  glewPrintExt("EGL_ANGLE_d3d_share_handle_client_buffer", EGLEW_ANGLE_d3d_share_handle_client_buffer, eglewIsSupported("EGL_ANGLE_d3d_share_handle_client_buffer"), eglewGetExtension("EGL_ANGLE_d3d_share_handle_client_buffer"));
}

#endif /* EGL_ANGLE_d3d_share_handle_client_buffer */

#ifdef EGL_ANGLE_device_d3d

static void _glewInfo_EGL_ANGLE_device_d3d (void)
{
  glewPrintExt("EGL_ANGLE_device_d3d", EGLEW_ANGLE_device_d3d, eglewIsSupported("EGL_ANGLE_device_d3d"), eglewGetExtension("EGL_ANGLE_device_d3d"));
}

#endif /* EGL_ANGLE_device_d3d */

#ifdef EGL_ANGLE_query_surface_pointer

static void _glewInfo_EGL_ANGLE_query_surface_pointer (void)
{
  GLboolean fi = glewPrintExt("EGL_ANGLE_query_surface_pointer", EGLEW_ANGLE_query_surface_pointer, eglewIsSupported("EGL_ANGLE_query_surface_pointer"), eglewGetExtension("EGL_ANGLE_query_surface_pointer"));

  glewInfoFunc(fi, "eglQuerySurfacePointerANGLE", eglQuerySurfacePointerANGLE == NULL);
}

#endif /* EGL_ANGLE_query_surface_pointer */

#ifdef EGL_ANGLE_surface_d3d_texture_2d_share_handle

static void _glewInfo_EGL_ANGLE_surface_d3d_texture_2d_share_handle (void)
{
  glewPrintExt("EGL_ANGLE_surface_d3d_texture_2d_share_handle", EGLEW_ANGLE_surface_d3d_texture_2d_share_handle, eglewIsSupported("EGL_ANGLE_surface_d3d_texture_2d_share_handle"), eglewGetExtension("EGL_ANGLE_surface_d3d_texture_2d_share_handle"));
}

#endif /* EGL_ANGLE_surface_d3d_texture_2d_share_handle */

#ifdef EGL_ANGLE_window_fixed_size

static void _glewInfo_EGL_ANGLE_window_fixed_size (void)
{
  glewPrintExt("EGL_ANGLE_window_fixed_size", EGLEW_ANGLE_window_fixed_size, eglewIsSupported("EGL_ANGLE_window_fixed_size"), eglewGetExtension("EGL_ANGLE_window_fixed_size"));
}

#endif /* EGL_ANGLE_window_fixed_size */

#ifdef EGL_ARM_image_format

static void _glewInfo_EGL_ARM_image_format (void)
{
  glewPrintExt("EGL_ARM_image_format", EGLEW_ARM_image_format, eglewIsSupported("EGL_ARM_image_format"), eglewGetExtension("EGL_ARM_image_format"));
}

#endif /* EGL_ARM_image_format */

#ifdef EGL_ARM_implicit_external_sync

static void _glewInfo_EGL_ARM_implicit_external_sync (void)
{
  glewPrintExt("EGL_ARM_implicit_external_sync", EGLEW_ARM_implicit_external_sync, eglewIsSupported("EGL_ARM_implicit_external_sync"), eglewGetExtension("EGL_ARM_implicit_external_sync"));
}

#endif /* EGL_ARM_implicit_external_sync */

#ifdef EGL_ARM_pixmap_multisample_discard

static void _glewInfo_EGL_ARM_pixmap_multisample_discard (void)
{
  glewPrintExt("EGL_ARM_pixmap_multisample_discard", EGLEW_ARM_pixmap_multisample_discard, eglewIsSupported("EGL_ARM_pixmap_multisample_discard"), eglewGetExtension("EGL_ARM_pixmap_multisample_discard"));
}

#endif /* EGL_ARM_pixmap_multisample_discard */

#ifdef EGL_EXT_bind_to_front

static void _glewInfo_EGL_EXT_bind_to_front (void)
{
  glewPrintExt("EGL_EXT_bind_to_front", EGLEW_EXT_bind_to_front, eglewIsSupported("EGL_EXT_bind_to_front"), eglewGetExtension("EGL_EXT_bind_to_front"));
}

#endif /* EGL_EXT_bind_to_front */

#ifdef EGL_EXT_buffer_age

static void _glewInfo_EGL_EXT_buffer_age (void)
{
  glewPrintExt("EGL_EXT_buffer_age", EGLEW_EXT_buffer_age, eglewIsSupported("EGL_EXT_buffer_age"), eglewGetExtension("EGL_EXT_buffer_age"));
}

#endif /* EGL_EXT_buffer_age */

#ifdef EGL_EXT_client_extensions

static void _glewInfo_EGL_EXT_client_extensions (void)
{
  glewPrintExt("EGL_EXT_client_extensions", EGLEW_EXT_client_extensions, eglewIsSupported("EGL_EXT_client_extensions"), eglewGetExtension("EGL_EXT_client_extensions"));
}

#endif /* EGL_EXT_client_extensions */

#ifdef EGL_EXT_client_sync

static void _glewInfo_EGL_EXT_client_sync (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_client_sync", EGLEW_EXT_client_sync, eglewIsSupported("EGL_EXT_client_sync"), eglewGetExtension("EGL_EXT_client_sync"));

  glewInfoFunc(fi, "eglClientSignalSyncEXT", eglClientSignalSyncEXT == NULL);
}

#endif /* EGL_EXT_client_sync */

#ifdef EGL_EXT_compositor

static void _glewInfo_EGL_EXT_compositor (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_compositor", EGLEW_EXT_compositor, eglewIsSupported("EGL_EXT_compositor"), eglewGetExtension("EGL_EXT_compositor"));

  glewInfoFunc(fi, "eglCompositorBindTexWindowEXT", eglCompositorBindTexWindowEXT == NULL);
  glewInfoFunc(fi, "eglCompositorSetContextAttributesEXT", eglCompositorSetContextAttributesEXT == NULL);
  glewInfoFunc(fi, "eglCompositorSetContextListEXT", eglCompositorSetContextListEXT == NULL);
  glewInfoFunc(fi, "eglCompositorSetSizeEXT", eglCompositorSetSizeEXT == NULL);
  glewInfoFunc(fi, "eglCompositorSetWindowAttributesEXT", eglCompositorSetWindowAttributesEXT == NULL);
  glewInfoFunc(fi, "eglCompositorSetWindowListEXT", eglCompositorSetWindowListEXT == NULL);
  glewInfoFunc(fi, "eglCompositorSwapPolicyEXT", eglCompositorSwapPolicyEXT == NULL);
}

#endif /* EGL_EXT_compositor */

#ifdef EGL_EXT_create_context_robustness

static void _glewInfo_EGL_EXT_create_context_robustness (void)
{
  glewPrintExt("EGL_EXT_create_context_robustness", EGLEW_EXT_create_context_robustness, eglewIsSupported("EGL_EXT_create_context_robustness"), eglewGetExtension("EGL_EXT_create_context_robustness"));
}

#endif /* EGL_EXT_create_context_robustness */

#ifdef EGL_EXT_device_base

static void _glewInfo_EGL_EXT_device_base (void)
{
  glewPrintExt("EGL_EXT_device_base", EGLEW_EXT_device_base, eglewIsSupported("EGL_EXT_device_base"), eglewGetExtension("EGL_EXT_device_base"));
}

#endif /* EGL_EXT_device_base */

#ifdef EGL_EXT_device_drm

static void _glewInfo_EGL_EXT_device_drm (void)
{
  glewPrintExt("EGL_EXT_device_drm", EGLEW_EXT_device_drm, eglewIsSupported("EGL_EXT_device_drm"), eglewGetExtension("EGL_EXT_device_drm"));
}

#endif /* EGL_EXT_device_drm */

#ifdef EGL_EXT_device_enumeration

static void _glewInfo_EGL_EXT_device_enumeration (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_device_enumeration", EGLEW_EXT_device_enumeration, eglewIsSupported("EGL_EXT_device_enumeration"), eglewGetExtension("EGL_EXT_device_enumeration"));

  glewInfoFunc(fi, "eglQueryDevicesEXT", eglQueryDevicesEXT == NULL);
}

#endif /* EGL_EXT_device_enumeration */

#ifdef EGL_EXT_device_openwf

static void _glewInfo_EGL_EXT_device_openwf (void)
{
  glewPrintExt("EGL_EXT_device_openwf", EGLEW_EXT_device_openwf, eglewIsSupported("EGL_EXT_device_openwf"), eglewGetExtension("EGL_EXT_device_openwf"));
}

#endif /* EGL_EXT_device_openwf */

#ifdef EGL_EXT_device_query

static void _glewInfo_EGL_EXT_device_query (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_device_query", EGLEW_EXT_device_query, eglewIsSupported("EGL_EXT_device_query"), eglewGetExtension("EGL_EXT_device_query"));

  glewInfoFunc(fi, "eglQueryDeviceAttribEXT", eglQueryDeviceAttribEXT == NULL);
  glewInfoFunc(fi, "eglQueryDeviceStringEXT", eglQueryDeviceStringEXT == NULL);
  glewInfoFunc(fi, "eglQueryDisplayAttribEXT", eglQueryDisplayAttribEXT == NULL);
}

#endif /* EGL_EXT_device_query */

#ifdef EGL_EXT_gl_colorspace_bt2020_linear

static void _glewInfo_EGL_EXT_gl_colorspace_bt2020_linear (void)
{
  glewPrintExt("EGL_EXT_gl_colorspace_bt2020_linear", EGLEW_EXT_gl_colorspace_bt2020_linear, eglewIsSupported("EGL_EXT_gl_colorspace_bt2020_linear"), eglewGetExtension("EGL_EXT_gl_colorspace_bt2020_linear"));
}

#endif /* EGL_EXT_gl_colorspace_bt2020_linear */

#ifdef EGL_EXT_gl_colorspace_bt2020_pq

static void _glewInfo_EGL_EXT_gl_colorspace_bt2020_pq (void)
{
  glewPrintExt("EGL_EXT_gl_colorspace_bt2020_pq", EGLEW_EXT_gl_colorspace_bt2020_pq, eglewIsSupported("EGL_EXT_gl_colorspace_bt2020_pq"), eglewGetExtension("EGL_EXT_gl_colorspace_bt2020_pq"));
}

#endif /* EGL_EXT_gl_colorspace_bt2020_pq */

#ifdef EGL_EXT_gl_colorspace_display_p3

static void _glewInfo_EGL_EXT_gl_colorspace_display_p3 (void)
{
  glewPrintExt("EGL_EXT_gl_colorspace_display_p3", EGLEW_EXT_gl_colorspace_display_p3, eglewIsSupported("EGL_EXT_gl_colorspace_display_p3"), eglewGetExtension("EGL_EXT_gl_colorspace_display_p3"));
}

#endif /* EGL_EXT_gl_colorspace_display_p3 */

#ifdef EGL_EXT_gl_colorspace_display_p3_linear

static void _glewInfo_EGL_EXT_gl_colorspace_display_p3_linear (void)
{
  glewPrintExt("EGL_EXT_gl_colorspace_display_p3_linear", EGLEW_EXT_gl_colorspace_display_p3_linear, eglewIsSupported("EGL_EXT_gl_colorspace_display_p3_linear"), eglewGetExtension("EGL_EXT_gl_colorspace_display_p3_linear"));
}

#endif /* EGL_EXT_gl_colorspace_display_p3_linear */

#ifdef EGL_EXT_gl_colorspace_display_p3_passthrough

static void _glewInfo_EGL_EXT_gl_colorspace_display_p3_passthrough (void)
{
  glewPrintExt("EGL_EXT_gl_colorspace_display_p3_passthrough", EGLEW_EXT_gl_colorspace_display_p3_passthrough, eglewIsSupported("EGL_EXT_gl_colorspace_display_p3_passthrough"), eglewGetExtension("EGL_EXT_gl_colorspace_display_p3_passthrough"));
}

#endif /* EGL_EXT_gl_colorspace_display_p3_passthrough */

#ifdef EGL_EXT_gl_colorspace_scrgb

static void _glewInfo_EGL_EXT_gl_colorspace_scrgb (void)
{
  glewPrintExt("EGL_EXT_gl_colorspace_scrgb", EGLEW_EXT_gl_colorspace_scrgb, eglewIsSupported("EGL_EXT_gl_colorspace_scrgb"), eglewGetExtension("EGL_EXT_gl_colorspace_scrgb"));
}

#endif /* EGL_EXT_gl_colorspace_scrgb */

#ifdef EGL_EXT_gl_colorspace_scrgb_linear

static void _glewInfo_EGL_EXT_gl_colorspace_scrgb_linear (void)
{
  glewPrintExt("EGL_EXT_gl_colorspace_scrgb_linear", EGLEW_EXT_gl_colorspace_scrgb_linear, eglewIsSupported("EGL_EXT_gl_colorspace_scrgb_linear"), eglewGetExtension("EGL_EXT_gl_colorspace_scrgb_linear"));
}

#endif /* EGL_EXT_gl_colorspace_scrgb_linear */

#ifdef EGL_EXT_image_dma_buf_import

static void _glewInfo_EGL_EXT_image_dma_buf_import (void)
{
  glewPrintExt("EGL_EXT_image_dma_buf_import", EGLEW_EXT_image_dma_buf_import, eglewIsSupported("EGL_EXT_image_dma_buf_import"), eglewGetExtension("EGL_EXT_image_dma_buf_import"));
}

#endif /* EGL_EXT_image_dma_buf_import */

#ifdef EGL_EXT_image_dma_buf_import_modifiers

static void _glewInfo_EGL_EXT_image_dma_buf_import_modifiers (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_image_dma_buf_import_modifiers", EGLEW_EXT_image_dma_buf_import_modifiers, eglewIsSupported("EGL_EXT_image_dma_buf_import_modifiers"), eglewGetExtension("EGL_EXT_image_dma_buf_import_modifiers"));

  glewInfoFunc(fi, "eglQueryDmaBufFormatsEXT", eglQueryDmaBufFormatsEXT == NULL);
  glewInfoFunc(fi, "eglQueryDmaBufModifiersEXT", eglQueryDmaBufModifiersEXT == NULL);
}

#endif /* EGL_EXT_image_dma_buf_import_modifiers */

#ifdef EGL_EXT_image_gl_colorspace

static void _glewInfo_EGL_EXT_image_gl_colorspace (void)
{
  glewPrintExt("EGL_EXT_image_gl_colorspace", EGLEW_EXT_image_gl_colorspace, eglewIsSupported("EGL_EXT_image_gl_colorspace"), eglewGetExtension("EGL_EXT_image_gl_colorspace"));
}

#endif /* EGL_EXT_image_gl_colorspace */

#ifdef EGL_EXT_image_implicit_sync_control

static void _glewInfo_EGL_EXT_image_implicit_sync_control (void)
{
  glewPrintExt("EGL_EXT_image_implicit_sync_control", EGLEW_EXT_image_implicit_sync_control, eglewIsSupported("EGL_EXT_image_implicit_sync_control"), eglewGetExtension("EGL_EXT_image_implicit_sync_control"));
}

#endif /* EGL_EXT_image_implicit_sync_control */

#ifdef EGL_EXT_multiview_window

static void _glewInfo_EGL_EXT_multiview_window (void)
{
  glewPrintExt("EGL_EXT_multiview_window", EGLEW_EXT_multiview_window, eglewIsSupported("EGL_EXT_multiview_window"), eglewGetExtension("EGL_EXT_multiview_window"));
}

#endif /* EGL_EXT_multiview_window */

#ifdef EGL_EXT_output_base

static void _glewInfo_EGL_EXT_output_base (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_output_base", EGLEW_EXT_output_base, eglewIsSupported("EGL_EXT_output_base"), eglewGetExtension("EGL_EXT_output_base"));

  glewInfoFunc(fi, "eglGetOutputLayersEXT", eglGetOutputLayersEXT == NULL);
  glewInfoFunc(fi, "eglGetOutputPortsEXT", eglGetOutputPortsEXT == NULL);
  glewInfoFunc(fi, "eglOutputLayerAttribEXT", eglOutputLayerAttribEXT == NULL);
  glewInfoFunc(fi, "eglOutputPortAttribEXT", eglOutputPortAttribEXT == NULL);
  glewInfoFunc(fi, "eglQueryOutputLayerAttribEXT", eglQueryOutputLayerAttribEXT == NULL);
  glewInfoFunc(fi, "eglQueryOutputLayerStringEXT", eglQueryOutputLayerStringEXT == NULL);
  glewInfoFunc(fi, "eglQueryOutputPortAttribEXT", eglQueryOutputPortAttribEXT == NULL);
  glewInfoFunc(fi, "eglQueryOutputPortStringEXT", eglQueryOutputPortStringEXT == NULL);
}

#endif /* EGL_EXT_output_base */

#ifdef EGL_EXT_output_drm

static void _glewInfo_EGL_EXT_output_drm (void)
{
  glewPrintExt("EGL_EXT_output_drm", EGLEW_EXT_output_drm, eglewIsSupported("EGL_EXT_output_drm"), eglewGetExtension("EGL_EXT_output_drm"));
}

#endif /* EGL_EXT_output_drm */

#ifdef EGL_EXT_output_openwf

static void _glewInfo_EGL_EXT_output_openwf (void)
{
  glewPrintExt("EGL_EXT_output_openwf", EGLEW_EXT_output_openwf, eglewIsSupported("EGL_EXT_output_openwf"), eglewGetExtension("EGL_EXT_output_openwf"));
}

#endif /* EGL_EXT_output_openwf */

#ifdef EGL_EXT_pixel_format_float

static void _glewInfo_EGL_EXT_pixel_format_float (void)
{
  glewPrintExt("EGL_EXT_pixel_format_float", EGLEW_EXT_pixel_format_float, eglewIsSupported("EGL_EXT_pixel_format_float"), eglewGetExtension("EGL_EXT_pixel_format_float"));
}

#endif /* EGL_EXT_pixel_format_float */

#ifdef EGL_EXT_platform_base

static void _glewInfo_EGL_EXT_platform_base (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_platform_base", EGLEW_EXT_platform_base, eglewIsSupported("EGL_EXT_platform_base"), eglewGetExtension("EGL_EXT_platform_base"));

  glewInfoFunc(fi, "eglCreatePlatformPixmapSurfaceEXT", eglCreatePlatformPixmapSurfaceEXT == NULL);
  glewInfoFunc(fi, "eglCreatePlatformWindowSurfaceEXT", eglCreatePlatformWindowSurfaceEXT == NULL);
  glewInfoFunc(fi, "eglGetPlatformDisplayEXT", eglGetPlatformDisplayEXT == NULL);
}

#endif /* EGL_EXT_platform_base */

#ifdef EGL_EXT_platform_device

static void _glewInfo_EGL_EXT_platform_device (void)
{
  glewPrintExt("EGL_EXT_platform_device", EGLEW_EXT_platform_device, eglewIsSupported("EGL_EXT_platform_device"), eglewGetExtension("EGL_EXT_platform_device"));
}

#endif /* EGL_EXT_platform_device */

#ifdef EGL_EXT_platform_wayland

static void _glewInfo_EGL_EXT_platform_wayland (void)
{
  glewPrintExt("EGL_EXT_platform_wayland", EGLEW_EXT_platform_wayland, eglewIsSupported("EGL_EXT_platform_wayland"), eglewGetExtension("EGL_EXT_platform_wayland"));
}

#endif /* EGL_EXT_platform_wayland */

#ifdef EGL_EXT_platform_x11

static void _glewInfo_EGL_EXT_platform_x11 (void)
{
  glewPrintExt("EGL_EXT_platform_x11", EGLEW_EXT_platform_x11, eglewIsSupported("EGL_EXT_platform_x11"), eglewGetExtension("EGL_EXT_platform_x11"));
}

#endif /* EGL_EXT_platform_x11 */

#ifdef EGL_EXT_protected_content

static void _glewInfo_EGL_EXT_protected_content (void)
{
  glewPrintExt("EGL_EXT_protected_content", EGLEW_EXT_protected_content, eglewIsSupported("EGL_EXT_protected_content"), eglewGetExtension("EGL_EXT_protected_content"));
}

#endif /* EGL_EXT_protected_content */

#ifdef EGL_EXT_protected_surface

static void _glewInfo_EGL_EXT_protected_surface (void)
{
  glewPrintExt("EGL_EXT_protected_surface", EGLEW_EXT_protected_surface, eglewIsSupported("EGL_EXT_protected_surface"), eglewGetExtension("EGL_EXT_protected_surface"));
}

#endif /* EGL_EXT_protected_surface */

#ifdef EGL_EXT_stream_consumer_egloutput

static void _glewInfo_EGL_EXT_stream_consumer_egloutput (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_stream_consumer_egloutput", EGLEW_EXT_stream_consumer_egloutput, eglewIsSupported("EGL_EXT_stream_consumer_egloutput"), eglewGetExtension("EGL_EXT_stream_consumer_egloutput"));

  glewInfoFunc(fi, "eglStreamConsumerOutputEXT", eglStreamConsumerOutputEXT == NULL);
}

#endif /* EGL_EXT_stream_consumer_egloutput */

#ifdef EGL_EXT_surface_CTA861_3_metadata

static void _glewInfo_EGL_EXT_surface_CTA861_3_metadata (void)
{
  glewPrintExt("EGL_EXT_surface_CTA861_3_metadata", EGLEW_EXT_surface_CTA861_3_metadata, eglewIsSupported("EGL_EXT_surface_CTA861_3_metadata"), eglewGetExtension("EGL_EXT_surface_CTA861_3_metadata"));
}

#endif /* EGL_EXT_surface_CTA861_3_metadata */

#ifdef EGL_EXT_surface_SMPTE2086_metadata

static void _glewInfo_EGL_EXT_surface_SMPTE2086_metadata (void)
{
  glewPrintExt("EGL_EXT_surface_SMPTE2086_metadata", EGLEW_EXT_surface_SMPTE2086_metadata, eglewIsSupported("EGL_EXT_surface_SMPTE2086_metadata"), eglewGetExtension("EGL_EXT_surface_SMPTE2086_metadata"));
}

#endif /* EGL_EXT_surface_SMPTE2086_metadata */

#ifdef EGL_EXT_swap_buffers_with_damage

static void _glewInfo_EGL_EXT_swap_buffers_with_damage (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_swap_buffers_with_damage", EGLEW_EXT_swap_buffers_with_damage, eglewIsSupported("EGL_EXT_swap_buffers_with_damage"), eglewGetExtension("EGL_EXT_swap_buffers_with_damage"));

  glewInfoFunc(fi, "eglSwapBuffersWithDamageEXT", eglSwapBuffersWithDamageEXT == NULL);
}

#endif /* EGL_EXT_swap_buffers_with_damage */

#ifdef EGL_EXT_sync_reuse

static void _glewInfo_EGL_EXT_sync_reuse (void)
{
  GLboolean fi = glewPrintExt("EGL_EXT_sync_reuse", EGLEW_EXT_sync_reuse, eglewIsSupported("EGL_EXT_sync_reuse"), eglewGetExtension("EGL_EXT_sync_reuse"));

  glewInfoFunc(fi, "eglUnsignalSyncEXT", eglUnsignalSyncEXT == NULL);
}

#endif /* EGL_EXT_sync_reuse */

#ifdef EGL_EXT_yuv_surface

static void _glewInfo_EGL_EXT_yuv_surface (void)
{
  glewPrintExt("EGL_EXT_yuv_surface", EGLEW_EXT_yuv_surface, eglewIsSupported("EGL_EXT_yuv_surface"), eglewGetExtension("EGL_EXT_yuv_surface"));
}

#endif /* EGL_EXT_yuv_surface */

#ifdef EGL_HI_clientpixmap

static void _glewInfo_EGL_HI_clientpixmap (void)
{
  GLboolean fi = glewPrintExt("EGL_HI_clientpixmap", EGLEW_HI_clientpixmap, eglewIsSupported("EGL_HI_clientpixmap"), eglewGetExtension("EGL_HI_clientpixmap"));

  glewInfoFunc(fi, "eglCreatePixmapSurfaceHI", eglCreatePixmapSurfaceHI == NULL);
}

#endif /* EGL_HI_clientpixmap */

#ifdef EGL_HI_colorformats

static void _glewInfo_EGL_HI_colorformats (void)
{
  glewPrintExt("EGL_HI_colorformats", EGLEW_HI_colorformats, eglewIsSupported("EGL_HI_colorformats"), eglewGetExtension("EGL_HI_colorformats"));
}

#endif /* EGL_HI_colorformats */

#ifdef EGL_IMG_context_priority

static void _glewInfo_EGL_IMG_context_priority (void)
{
  glewPrintExt("EGL_IMG_context_priority", EGLEW_IMG_context_priority, eglewIsSupported("EGL_IMG_context_priority"), eglewGetExtension("EGL_IMG_context_priority"));
}

#endif /* EGL_IMG_context_priority */

#ifdef EGL_IMG_image_plane_attribs

static void _glewInfo_EGL_IMG_image_plane_attribs (void)
{
  glewPrintExt("EGL_IMG_image_plane_attribs", EGLEW_IMG_image_plane_attribs, eglewIsSupported("EGL_IMG_image_plane_attribs"), eglewGetExtension("EGL_IMG_image_plane_attribs"));
}

#endif /* EGL_IMG_image_plane_attribs */

#ifdef EGL_KHR_cl_event

static void _glewInfo_EGL_KHR_cl_event (void)
{
  glewPrintExt("EGL_KHR_cl_event", EGLEW_KHR_cl_event, eglewIsSupported("EGL_KHR_cl_event"), eglewGetExtension("EGL_KHR_cl_event"));
}

#endif /* EGL_KHR_cl_event */

#ifdef EGL_KHR_cl_event2

static void _glewInfo_EGL_KHR_cl_event2 (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_cl_event2", EGLEW_KHR_cl_event2, eglewIsSupported("EGL_KHR_cl_event2"), eglewGetExtension("EGL_KHR_cl_event2"));

  glewInfoFunc(fi, "eglCreateSync64KHR", eglCreateSync64KHR == NULL);
}

#endif /* EGL_KHR_cl_event2 */

#ifdef EGL_KHR_client_get_all_proc_addresses

static void _glewInfo_EGL_KHR_client_get_all_proc_addresses (void)
{
  glewPrintExt("EGL_KHR_client_get_all_proc_addresses", EGLEW_KHR_client_get_all_proc_addresses, eglewIsSupported("EGL_KHR_client_get_all_proc_addresses"), eglewGetExtension("EGL_KHR_client_get_all_proc_addresses"));
}

#endif /* EGL_KHR_client_get_all_proc_addresses */

#ifdef EGL_KHR_config_attribs

static void _glewInfo_EGL_KHR_config_attribs (void)
{
  glewPrintExt("EGL_KHR_config_attribs", EGLEW_KHR_config_attribs, eglewIsSupported("EGL_KHR_config_attribs"), eglewGetExtension("EGL_KHR_config_attribs"));
}

#endif /* EGL_KHR_config_attribs */

#ifdef EGL_KHR_context_flush_control

static void _glewInfo_EGL_KHR_context_flush_control (void)
{
  glewPrintExt("EGL_KHR_context_flush_control", EGLEW_KHR_context_flush_control, eglewIsSupported("EGL_KHR_context_flush_control"), eglewGetExtension("EGL_KHR_context_flush_control"));
}

#endif /* EGL_KHR_context_flush_control */

#ifdef EGL_KHR_create_context

static void _glewInfo_EGL_KHR_create_context (void)
{
  glewPrintExt("EGL_KHR_create_context", EGLEW_KHR_create_context, eglewIsSupported("EGL_KHR_create_context"), eglewGetExtension("EGL_KHR_create_context"));
}

#endif /* EGL_KHR_create_context */

#ifdef EGL_KHR_create_context_no_error

static void _glewInfo_EGL_KHR_create_context_no_error (void)
{
  glewPrintExt("EGL_KHR_create_context_no_error", EGLEW_KHR_create_context_no_error, eglewIsSupported("EGL_KHR_create_context_no_error"), eglewGetExtension("EGL_KHR_create_context_no_error"));
}

#endif /* EGL_KHR_create_context_no_error */

#ifdef EGL_KHR_debug

static void _glewInfo_EGL_KHR_debug (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_debug", EGLEW_KHR_debug, eglewIsSupported("EGL_KHR_debug"), eglewGetExtension("EGL_KHR_debug"));

  glewInfoFunc(fi, "eglDebugMessageControlKHR", eglDebugMessageControlKHR == NULL);
  glewInfoFunc(fi, "eglLabelObjectKHR", eglLabelObjectKHR == NULL);
  glewInfoFunc(fi, "eglQueryDebugKHR", eglQueryDebugKHR == NULL);
}

#endif /* EGL_KHR_debug */

#ifdef EGL_KHR_display_reference

static void _glewInfo_EGL_KHR_display_reference (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_display_reference", EGLEW_KHR_display_reference, eglewIsSupported("EGL_KHR_display_reference"), eglewGetExtension("EGL_KHR_display_reference"));

  glewInfoFunc(fi, "eglQueryDisplayAttribKHR", eglQueryDisplayAttribKHR == NULL);
}

#endif /* EGL_KHR_display_reference */

#ifdef EGL_KHR_fence_sync

static void _glewInfo_EGL_KHR_fence_sync (void)
{
  glewPrintExt("EGL_KHR_fence_sync", EGLEW_KHR_fence_sync, eglewIsSupported("EGL_KHR_fence_sync"), eglewGetExtension("EGL_KHR_fence_sync"));
}

#endif /* EGL_KHR_fence_sync */

#ifdef EGL_KHR_get_all_proc_addresses

static void _glewInfo_EGL_KHR_get_all_proc_addresses (void)
{
  glewPrintExt("EGL_KHR_get_all_proc_addresses", EGLEW_KHR_get_all_proc_addresses, eglewIsSupported("EGL_KHR_get_all_proc_addresses"), eglewGetExtension("EGL_KHR_get_all_proc_addresses"));
}

#endif /* EGL_KHR_get_all_proc_addresses */

#ifdef EGL_KHR_gl_colorspace

static void _glewInfo_EGL_KHR_gl_colorspace (void)
{
  glewPrintExt("EGL_KHR_gl_colorspace", EGLEW_KHR_gl_colorspace, eglewIsSupported("EGL_KHR_gl_colorspace"), eglewGetExtension("EGL_KHR_gl_colorspace"));
}

#endif /* EGL_KHR_gl_colorspace */

#ifdef EGL_KHR_gl_renderbuffer_image

static void _glewInfo_EGL_KHR_gl_renderbuffer_image (void)
{
  glewPrintExt("EGL_KHR_gl_renderbuffer_image", EGLEW_KHR_gl_renderbuffer_image, eglewIsSupported("EGL_KHR_gl_renderbuffer_image"), eglewGetExtension("EGL_KHR_gl_renderbuffer_image"));
}

#endif /* EGL_KHR_gl_renderbuffer_image */

#ifdef EGL_KHR_gl_texture_2D_image

static void _glewInfo_EGL_KHR_gl_texture_2D_image (void)
{
  glewPrintExt("EGL_KHR_gl_texture_2D_image", EGLEW_KHR_gl_texture_2D_image, eglewIsSupported("EGL_KHR_gl_texture_2D_image"), eglewGetExtension("EGL_KHR_gl_texture_2D_image"));
}

#endif /* EGL_KHR_gl_texture_2D_image */

#ifdef EGL_KHR_gl_texture_3D_image

static void _glewInfo_EGL_KHR_gl_texture_3D_image (void)
{
  glewPrintExt("EGL_KHR_gl_texture_3D_image", EGLEW_KHR_gl_texture_3D_image, eglewIsSupported("EGL_KHR_gl_texture_3D_image"), eglewGetExtension("EGL_KHR_gl_texture_3D_image"));
}

#endif /* EGL_KHR_gl_texture_3D_image */

#ifdef EGL_KHR_gl_texture_cubemap_image

static void _glewInfo_EGL_KHR_gl_texture_cubemap_image (void)
{
  glewPrintExt("EGL_KHR_gl_texture_cubemap_image", EGLEW_KHR_gl_texture_cubemap_image, eglewIsSupported("EGL_KHR_gl_texture_cubemap_image"), eglewGetExtension("EGL_KHR_gl_texture_cubemap_image"));
}

#endif /* EGL_KHR_gl_texture_cubemap_image */

#ifdef EGL_KHR_image

static void _glewInfo_EGL_KHR_image (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_image", EGLEW_KHR_image, eglewIsSupported("EGL_KHR_image"), eglewGetExtension("EGL_KHR_image"));

  glewInfoFunc(fi, "eglCreateImageKHR", eglCreateImageKHR == NULL);
  glewInfoFunc(fi, "eglDestroyImageKHR", eglDestroyImageKHR == NULL);
}

#endif /* EGL_KHR_image */

#ifdef EGL_KHR_image_base

static void _glewInfo_EGL_KHR_image_base (void)
{
  glewPrintExt("EGL_KHR_image_base", EGLEW_KHR_image_base, eglewIsSupported("EGL_KHR_image_base"), eglewGetExtension("EGL_KHR_image_base"));
}

#endif /* EGL_KHR_image_base */

#ifdef EGL_KHR_image_pixmap

static void _glewInfo_EGL_KHR_image_pixmap (void)
{
  glewPrintExt("EGL_KHR_image_pixmap", EGLEW_KHR_image_pixmap, eglewIsSupported("EGL_KHR_image_pixmap"), eglewGetExtension("EGL_KHR_image_pixmap"));
}

#endif /* EGL_KHR_image_pixmap */

#ifdef EGL_KHR_lock_surface

static void _glewInfo_EGL_KHR_lock_surface (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_lock_surface", EGLEW_KHR_lock_surface, eglewIsSupported("EGL_KHR_lock_surface"), eglewGetExtension("EGL_KHR_lock_surface"));

  glewInfoFunc(fi, "eglLockSurfaceKHR", eglLockSurfaceKHR == NULL);
  glewInfoFunc(fi, "eglUnlockSurfaceKHR", eglUnlockSurfaceKHR == NULL);
}

#endif /* EGL_KHR_lock_surface */

#ifdef EGL_KHR_lock_surface2

static void _glewInfo_EGL_KHR_lock_surface2 (void)
{
  glewPrintExt("EGL_KHR_lock_surface2", EGLEW_KHR_lock_surface2, eglewIsSupported("EGL_KHR_lock_surface2"), eglewGetExtension("EGL_KHR_lock_surface2"));
}

#endif /* EGL_KHR_lock_surface2 */

#ifdef EGL_KHR_lock_surface3

static void _glewInfo_EGL_KHR_lock_surface3 (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_lock_surface3", EGLEW_KHR_lock_surface3, eglewIsSupported("EGL_KHR_lock_surface3"), eglewGetExtension("EGL_KHR_lock_surface3"));

  glewInfoFunc(fi, "eglQuerySurface64KHR", eglQuerySurface64KHR == NULL);
}

#endif /* EGL_KHR_lock_surface3 */

#ifdef EGL_KHR_mutable_render_buffer

static void _glewInfo_EGL_KHR_mutable_render_buffer (void)
{
  glewPrintExt("EGL_KHR_mutable_render_buffer", EGLEW_KHR_mutable_render_buffer, eglewIsSupported("EGL_KHR_mutable_render_buffer"), eglewGetExtension("EGL_KHR_mutable_render_buffer"));
}

#endif /* EGL_KHR_mutable_render_buffer */

#ifdef EGL_KHR_no_config_context

static void _glewInfo_EGL_KHR_no_config_context (void)
{
  glewPrintExt("EGL_KHR_no_config_context", EGLEW_KHR_no_config_context, eglewIsSupported("EGL_KHR_no_config_context"), eglewGetExtension("EGL_KHR_no_config_context"));
}

#endif /* EGL_KHR_no_config_context */

#ifdef EGL_KHR_partial_update

static void _glewInfo_EGL_KHR_partial_update (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_partial_update", EGLEW_KHR_partial_update, eglewIsSupported("EGL_KHR_partial_update"), eglewGetExtension("EGL_KHR_partial_update"));

  glewInfoFunc(fi, "eglSetDamageRegionKHR", eglSetDamageRegionKHR == NULL);
}

#endif /* EGL_KHR_partial_update */

#ifdef EGL_KHR_platform_android

static void _glewInfo_EGL_KHR_platform_android (void)
{
  glewPrintExt("EGL_KHR_platform_android", EGLEW_KHR_platform_android, eglewIsSupported("EGL_KHR_platform_android"), eglewGetExtension("EGL_KHR_platform_android"));
}

#endif /* EGL_KHR_platform_android */

#ifdef EGL_KHR_platform_gbm

static void _glewInfo_EGL_KHR_platform_gbm (void)
{
  glewPrintExt("EGL_KHR_platform_gbm", EGLEW_KHR_platform_gbm, eglewIsSupported("EGL_KHR_platform_gbm"), eglewGetExtension("EGL_KHR_platform_gbm"));
}

#endif /* EGL_KHR_platform_gbm */

#ifdef EGL_KHR_platform_wayland

static void _glewInfo_EGL_KHR_platform_wayland (void)
{
  glewPrintExt("EGL_KHR_platform_wayland", EGLEW_KHR_platform_wayland, eglewIsSupported("EGL_KHR_platform_wayland"), eglewGetExtension("EGL_KHR_platform_wayland"));
}

#endif /* EGL_KHR_platform_wayland */

#ifdef EGL_KHR_platform_x11

static void _glewInfo_EGL_KHR_platform_x11 (void)
{
  glewPrintExt("EGL_KHR_platform_x11", EGLEW_KHR_platform_x11, eglewIsSupported("EGL_KHR_platform_x11"), eglewGetExtension("EGL_KHR_platform_x11"));
}

#endif /* EGL_KHR_platform_x11 */

#ifdef EGL_KHR_reusable_sync

static void _glewInfo_EGL_KHR_reusable_sync (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_reusable_sync", EGLEW_KHR_reusable_sync, eglewIsSupported("EGL_KHR_reusable_sync"), eglewGetExtension("EGL_KHR_reusable_sync"));

  glewInfoFunc(fi, "eglClientWaitSyncKHR", eglClientWaitSyncKHR == NULL);
  glewInfoFunc(fi, "eglCreateSyncKHR", eglCreateSyncKHR == NULL);
  glewInfoFunc(fi, "eglDestroySyncKHR", eglDestroySyncKHR == NULL);
  glewInfoFunc(fi, "eglGetSyncAttribKHR", eglGetSyncAttribKHR == NULL);
  glewInfoFunc(fi, "eglSignalSyncKHR", eglSignalSyncKHR == NULL);
}

#endif /* EGL_KHR_reusable_sync */

#ifdef EGL_KHR_stream

static void _glewInfo_EGL_KHR_stream (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_stream", EGLEW_KHR_stream, eglewIsSupported("EGL_KHR_stream"), eglewGetExtension("EGL_KHR_stream"));

  glewInfoFunc(fi, "eglCreateStreamKHR", eglCreateStreamKHR == NULL);
  glewInfoFunc(fi, "eglDestroyStreamKHR", eglDestroyStreamKHR == NULL);
  glewInfoFunc(fi, "eglQueryStreamKHR", eglQueryStreamKHR == NULL);
  glewInfoFunc(fi, "eglQueryStreamu64KHR", eglQueryStreamu64KHR == NULL);
  glewInfoFunc(fi, "eglStreamAttribKHR", eglStreamAttribKHR == NULL);
}

#endif /* EGL_KHR_stream */

#ifdef EGL_KHR_stream_attrib

static void _glewInfo_EGL_KHR_stream_attrib (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_stream_attrib", EGLEW_KHR_stream_attrib, eglewIsSupported("EGL_KHR_stream_attrib"), eglewGetExtension("EGL_KHR_stream_attrib"));

  glewInfoFunc(fi, "eglCreateStreamAttribKHR", eglCreateStreamAttribKHR == NULL);
  glewInfoFunc(fi, "eglQueryStreamAttribKHR", eglQueryStreamAttribKHR == NULL);
  glewInfoFunc(fi, "eglSetStreamAttribKHR", eglSetStreamAttribKHR == NULL);
  glewInfoFunc(fi, "eglStreamConsumerAcquireAttribKHR", eglStreamConsumerAcquireAttribKHR == NULL);
  glewInfoFunc(fi, "eglStreamConsumerReleaseAttribKHR", eglStreamConsumerReleaseAttribKHR == NULL);
}

#endif /* EGL_KHR_stream_attrib */

#ifdef EGL_KHR_stream_consumer_gltexture

static void _glewInfo_EGL_KHR_stream_consumer_gltexture (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_stream_consumer_gltexture", EGLEW_KHR_stream_consumer_gltexture, eglewIsSupported("EGL_KHR_stream_consumer_gltexture"), eglewGetExtension("EGL_KHR_stream_consumer_gltexture"));

  glewInfoFunc(fi, "eglStreamConsumerAcquireKHR", eglStreamConsumerAcquireKHR == NULL);
  glewInfoFunc(fi, "eglStreamConsumerGLTextureExternalKHR", eglStreamConsumerGLTextureExternalKHR == NULL);
  glewInfoFunc(fi, "eglStreamConsumerReleaseKHR", eglStreamConsumerReleaseKHR == NULL);
}

#endif /* EGL_KHR_stream_consumer_gltexture */

#ifdef EGL_KHR_stream_cross_process_fd

static void _glewInfo_EGL_KHR_stream_cross_process_fd (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_stream_cross_process_fd", EGLEW_KHR_stream_cross_process_fd, eglewIsSupported("EGL_KHR_stream_cross_process_fd"), eglewGetExtension("EGL_KHR_stream_cross_process_fd"));

  glewInfoFunc(fi, "eglCreateStreamFromFileDescriptorKHR", eglCreateStreamFromFileDescriptorKHR == NULL);
  glewInfoFunc(fi, "eglGetStreamFileDescriptorKHR", eglGetStreamFileDescriptorKHR == NULL);
}

#endif /* EGL_KHR_stream_cross_process_fd */

#ifdef EGL_KHR_stream_fifo

static void _glewInfo_EGL_KHR_stream_fifo (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_stream_fifo", EGLEW_KHR_stream_fifo, eglewIsSupported("EGL_KHR_stream_fifo"), eglewGetExtension("EGL_KHR_stream_fifo"));

  glewInfoFunc(fi, "eglQueryStreamTimeKHR", eglQueryStreamTimeKHR == NULL);
}

#endif /* EGL_KHR_stream_fifo */

#ifdef EGL_KHR_stream_producer_aldatalocator

static void _glewInfo_EGL_KHR_stream_producer_aldatalocator (void)
{
  glewPrintExt("EGL_KHR_stream_producer_aldatalocator", EGLEW_KHR_stream_producer_aldatalocator, eglewIsSupported("EGL_KHR_stream_producer_aldatalocator"), eglewGetExtension("EGL_KHR_stream_producer_aldatalocator"));
}

#endif /* EGL_KHR_stream_producer_aldatalocator */

#ifdef EGL_KHR_stream_producer_eglsurface

static void _glewInfo_EGL_KHR_stream_producer_eglsurface (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_stream_producer_eglsurface", EGLEW_KHR_stream_producer_eglsurface, eglewIsSupported("EGL_KHR_stream_producer_eglsurface"), eglewGetExtension("EGL_KHR_stream_producer_eglsurface"));

  glewInfoFunc(fi, "eglCreateStreamProducerSurfaceKHR", eglCreateStreamProducerSurfaceKHR == NULL);
}

#endif /* EGL_KHR_stream_producer_eglsurface */

#ifdef EGL_KHR_surfaceless_context

static void _glewInfo_EGL_KHR_surfaceless_context (void)
{
  glewPrintExt("EGL_KHR_surfaceless_context", EGLEW_KHR_surfaceless_context, eglewIsSupported("EGL_KHR_surfaceless_context"), eglewGetExtension("EGL_KHR_surfaceless_context"));
}

#endif /* EGL_KHR_surfaceless_context */

#ifdef EGL_KHR_swap_buffers_with_damage

static void _glewInfo_EGL_KHR_swap_buffers_with_damage (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_swap_buffers_with_damage", EGLEW_KHR_swap_buffers_with_damage, eglewIsSupported("EGL_KHR_swap_buffers_with_damage"), eglewGetExtension("EGL_KHR_swap_buffers_with_damage"));

  glewInfoFunc(fi, "eglSwapBuffersWithDamageKHR", eglSwapBuffersWithDamageKHR == NULL);
}

#endif /* EGL_KHR_swap_buffers_with_damage */

#ifdef EGL_KHR_vg_parent_image

static void _glewInfo_EGL_KHR_vg_parent_image (void)
{
  glewPrintExt("EGL_KHR_vg_parent_image", EGLEW_KHR_vg_parent_image, eglewIsSupported("EGL_KHR_vg_parent_image"), eglewGetExtension("EGL_KHR_vg_parent_image"));
}

#endif /* EGL_KHR_vg_parent_image */

#ifdef EGL_KHR_wait_sync

static void _glewInfo_EGL_KHR_wait_sync (void)
{
  GLboolean fi = glewPrintExt("EGL_KHR_wait_sync", EGLEW_KHR_wait_sync, eglewIsSupported("EGL_KHR_wait_sync"), eglewGetExtension("EGL_KHR_wait_sync"));

  glewInfoFunc(fi, "eglWaitSyncKHR", eglWaitSyncKHR == NULL);
}

#endif /* EGL_KHR_wait_sync */

#ifdef EGL_MESA_drm_image

static void _glewInfo_EGL_MESA_drm_image (void)
{
  GLboolean fi = glewPrintExt("EGL_MESA_drm_image", EGLEW_MESA_drm_image, eglewIsSupported("EGL_MESA_drm_image"), eglewGetExtension("EGL_MESA_drm_image"));

  glewInfoFunc(fi, "eglCreateDRMImageMESA", eglCreateDRMImageMESA == NULL);
  glewInfoFunc(fi, "eglExportDRMImageMESA", eglExportDRMImageMESA == NULL);
}

#endif /* EGL_MESA_drm_image */

#ifdef EGL_MESA_image_dma_buf_export

static void _glewInfo_EGL_MESA_image_dma_buf_export (void)
{
  GLboolean fi = glewPrintExt("EGL_MESA_image_dma_buf_export", EGLEW_MESA_image_dma_buf_export, eglewIsSupported("EGL_MESA_image_dma_buf_export"), eglewGetExtension("EGL_MESA_image_dma_buf_export"));

  glewInfoFunc(fi, "eglExportDMABUFImageMESA", eglExportDMABUFImageMESA == NULL);
  glewInfoFunc(fi, "eglExportDMABUFImageQueryMESA", eglExportDMABUFImageQueryMESA == NULL);
}

#endif /* EGL_MESA_image_dma_buf_export */

#ifdef EGL_MESA_platform_gbm

static void _glewInfo_EGL_MESA_platform_gbm (void)
{
  glewPrintExt("EGL_MESA_platform_gbm", EGLEW_MESA_platform_gbm, eglewIsSupported("EGL_MESA_platform_gbm"), eglewGetExtension("EGL_MESA_platform_gbm"));
}

#endif /* EGL_MESA_platform_gbm */

#ifdef EGL_MESA_platform_surfaceless

static void _glewInfo_EGL_MESA_platform_surfaceless (void)
{
  glewPrintExt("EGL_MESA_platform_surfaceless", EGLEW_MESA_platform_surfaceless, eglewIsSupported("EGL_MESA_platform_surfaceless"), eglewGetExtension("EGL_MESA_platform_surfaceless"));
}

#endif /* EGL_MESA_platform_surfaceless */

#ifdef EGL_MESA_query_driver

static void _glewInfo_EGL_MESA_query_driver (void)
{
  GLboolean fi = glewPrintExt("EGL_MESA_query_driver", EGLEW_MESA_query_driver, eglewIsSupported("EGL_MESA_query_driver"), eglewGetExtension("EGL_MESA_query_driver"));

  glewInfoFunc(fi, "eglGetDisplayDriverConfig", eglGetDisplayDriverConfig == NULL);
  glewInfoFunc(fi, "eglGetDisplayDriverName", eglGetDisplayDriverName == NULL);
}

#endif /* EGL_MESA_query_driver */

#ifdef EGL_NOK_swap_region

static void _glewInfo_EGL_NOK_swap_region (void)
{
  GLboolean fi = glewPrintExt("EGL_NOK_swap_region", EGLEW_NOK_swap_region, eglewIsSupported("EGL_NOK_swap_region"), eglewGetExtension("EGL_NOK_swap_region"));

  glewInfoFunc(fi, "eglSwapBuffersRegionNOK", eglSwapBuffersRegionNOK == NULL);
}

#endif /* EGL_NOK_swap_region */

#ifdef EGL_NOK_swap_region2

static void _glewInfo_EGL_NOK_swap_region2 (void)
{
  GLboolean fi = glewPrintExt("EGL_NOK_swap_region2", EGLEW_NOK_swap_region2, eglewIsSupported("EGL_NOK_swap_region2"), eglewGetExtension("EGL_NOK_swap_region2"));

  glewInfoFunc(fi, "eglSwapBuffersRegion2NOK", eglSwapBuffersRegion2NOK == NULL);
}

#endif /* EGL_NOK_swap_region2 */

#ifdef EGL_NOK_texture_from_pixmap

static void _glewInfo_EGL_NOK_texture_from_pixmap (void)
{
  glewPrintExt("EGL_NOK_texture_from_pixmap", EGLEW_NOK_texture_from_pixmap, eglewIsSupported("EGL_NOK_texture_from_pixmap"), eglewGetExtension("EGL_NOK_texture_from_pixmap"));
}

#endif /* EGL_NOK_texture_from_pixmap */

#ifdef EGL_NV_3dvision_surface

static void _glewInfo_EGL_NV_3dvision_surface (void)
{
  glewPrintExt("EGL_NV_3dvision_surface", EGLEW_NV_3dvision_surface, eglewIsSupported("EGL_NV_3dvision_surface"), eglewGetExtension("EGL_NV_3dvision_surface"));
}

#endif /* EGL_NV_3dvision_surface */

#ifdef EGL_NV_context_priority_realtime

static void _glewInfo_EGL_NV_context_priority_realtime (void)
{
  glewPrintExt("EGL_NV_context_priority_realtime", EGLEW_NV_context_priority_realtime, eglewIsSupported("EGL_NV_context_priority_realtime"), eglewGetExtension("EGL_NV_context_priority_realtime"));
}

#endif /* EGL_NV_context_priority_realtime */

#ifdef EGL_NV_coverage_sample

static void _glewInfo_EGL_NV_coverage_sample (void)
{
  glewPrintExt("EGL_NV_coverage_sample", EGLEW_NV_coverage_sample, eglewIsSupported("EGL_NV_coverage_sample"), eglewGetExtension("EGL_NV_coverage_sample"));
}

#endif /* EGL_NV_coverage_sample */

#ifdef EGL_NV_coverage_sample_resolve

static void _glewInfo_EGL_NV_coverage_sample_resolve (void)
{
  glewPrintExt("EGL_NV_coverage_sample_resolve", EGLEW_NV_coverage_sample_resolve, eglewIsSupported("EGL_NV_coverage_sample_resolve"), eglewGetExtension("EGL_NV_coverage_sample_resolve"));
}

#endif /* EGL_NV_coverage_sample_resolve */

#ifdef EGL_NV_cuda_event

static void _glewInfo_EGL_NV_cuda_event (void)
{
  glewPrintExt("EGL_NV_cuda_event", EGLEW_NV_cuda_event, eglewIsSupported("EGL_NV_cuda_event"), eglewGetExtension("EGL_NV_cuda_event"));
}

#endif /* EGL_NV_cuda_event */

#ifdef EGL_NV_depth_nonlinear

static void _glewInfo_EGL_NV_depth_nonlinear (void)
{
  glewPrintExt("EGL_NV_depth_nonlinear", EGLEW_NV_depth_nonlinear, eglewIsSupported("EGL_NV_depth_nonlinear"), eglewGetExtension("EGL_NV_depth_nonlinear"));
}

#endif /* EGL_NV_depth_nonlinear */

#ifdef EGL_NV_device_cuda

static void _glewInfo_EGL_NV_device_cuda (void)
{
  glewPrintExt("EGL_NV_device_cuda", EGLEW_NV_device_cuda, eglewIsSupported("EGL_NV_device_cuda"), eglewGetExtension("EGL_NV_device_cuda"));
}

#endif /* EGL_NV_device_cuda */

#ifdef EGL_NV_native_query

static void _glewInfo_EGL_NV_native_query (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_native_query", EGLEW_NV_native_query, eglewIsSupported("EGL_NV_native_query"), eglewGetExtension("EGL_NV_native_query"));

  glewInfoFunc(fi, "eglQueryNativeDisplayNV", eglQueryNativeDisplayNV == NULL);
  glewInfoFunc(fi, "eglQueryNativePixmapNV", eglQueryNativePixmapNV == NULL);
  glewInfoFunc(fi, "eglQueryNativeWindowNV", eglQueryNativeWindowNV == NULL);
}

#endif /* EGL_NV_native_query */

#ifdef EGL_NV_post_convert_rounding

static void _glewInfo_EGL_NV_post_convert_rounding (void)
{
  glewPrintExt("EGL_NV_post_convert_rounding", EGLEW_NV_post_convert_rounding, eglewIsSupported("EGL_NV_post_convert_rounding"), eglewGetExtension("EGL_NV_post_convert_rounding"));
}

#endif /* EGL_NV_post_convert_rounding */

#ifdef EGL_NV_post_sub_buffer

static void _glewInfo_EGL_NV_post_sub_buffer (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_post_sub_buffer", EGLEW_NV_post_sub_buffer, eglewIsSupported("EGL_NV_post_sub_buffer"), eglewGetExtension("EGL_NV_post_sub_buffer"));

  glewInfoFunc(fi, "eglPostSubBufferNV", eglPostSubBufferNV == NULL);
}

#endif /* EGL_NV_post_sub_buffer */

#ifdef EGL_NV_quadruple_buffer

static void _glewInfo_EGL_NV_quadruple_buffer (void)
{
  glewPrintExt("EGL_NV_quadruple_buffer", EGLEW_NV_quadruple_buffer, eglewIsSupported("EGL_NV_quadruple_buffer"), eglewGetExtension("EGL_NV_quadruple_buffer"));
}

#endif /* EGL_NV_quadruple_buffer */

#ifdef EGL_NV_robustness_video_memory_purge

static void _glewInfo_EGL_NV_robustness_video_memory_purge (void)
{
  glewPrintExt("EGL_NV_robustness_video_memory_purge", EGLEW_NV_robustness_video_memory_purge, eglewIsSupported("EGL_NV_robustness_video_memory_purge"), eglewGetExtension("EGL_NV_robustness_video_memory_purge"));
}

#endif /* EGL_NV_robustness_video_memory_purge */

#ifdef EGL_NV_stream_consumer_gltexture_yuv

static void _glewInfo_EGL_NV_stream_consumer_gltexture_yuv (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_stream_consumer_gltexture_yuv", EGLEW_NV_stream_consumer_gltexture_yuv, eglewIsSupported("EGL_NV_stream_consumer_gltexture_yuv"), eglewGetExtension("EGL_NV_stream_consumer_gltexture_yuv"));

  glewInfoFunc(fi, "eglStreamConsumerGLTextureExternalAttribsNV", eglStreamConsumerGLTextureExternalAttribsNV == NULL);
}

#endif /* EGL_NV_stream_consumer_gltexture_yuv */

#ifdef EGL_NV_stream_cross_display

static void _glewInfo_EGL_NV_stream_cross_display (void)
{
  glewPrintExt("EGL_NV_stream_cross_display", EGLEW_NV_stream_cross_display, eglewIsSupported("EGL_NV_stream_cross_display"), eglewGetExtension("EGL_NV_stream_cross_display"));
}

#endif /* EGL_NV_stream_cross_display */

#ifdef EGL_NV_stream_cross_object

static void _glewInfo_EGL_NV_stream_cross_object (void)
{
  glewPrintExt("EGL_NV_stream_cross_object", EGLEW_NV_stream_cross_object, eglewIsSupported("EGL_NV_stream_cross_object"), eglewGetExtension("EGL_NV_stream_cross_object"));
}

#endif /* EGL_NV_stream_cross_object */

#ifdef EGL_NV_stream_cross_partition

static void _glewInfo_EGL_NV_stream_cross_partition (void)
{
  glewPrintExt("EGL_NV_stream_cross_partition", EGLEW_NV_stream_cross_partition, eglewIsSupported("EGL_NV_stream_cross_partition"), eglewGetExtension("EGL_NV_stream_cross_partition"));
}

#endif /* EGL_NV_stream_cross_partition */

#ifdef EGL_NV_stream_cross_process

static void _glewInfo_EGL_NV_stream_cross_process (void)
{
  glewPrintExt("EGL_NV_stream_cross_process", EGLEW_NV_stream_cross_process, eglewIsSupported("EGL_NV_stream_cross_process"), eglewGetExtension("EGL_NV_stream_cross_process"));
}

#endif /* EGL_NV_stream_cross_process */

#ifdef EGL_NV_stream_cross_system

static void _glewInfo_EGL_NV_stream_cross_system (void)
{
  glewPrintExt("EGL_NV_stream_cross_system", EGLEW_NV_stream_cross_system, eglewIsSupported("EGL_NV_stream_cross_system"), eglewGetExtension("EGL_NV_stream_cross_system"));
}

#endif /* EGL_NV_stream_cross_system */

#ifdef EGL_NV_stream_dma

static void _glewInfo_EGL_NV_stream_dma (void)
{
  glewPrintExt("EGL_NV_stream_dma", EGLEW_NV_stream_dma, eglewIsSupported("EGL_NV_stream_dma"), eglewGetExtension("EGL_NV_stream_dma"));
}

#endif /* EGL_NV_stream_dma */

#ifdef EGL_NV_stream_fifo_next

static void _glewInfo_EGL_NV_stream_fifo_next (void)
{
  glewPrintExt("EGL_NV_stream_fifo_next", EGLEW_NV_stream_fifo_next, eglewIsSupported("EGL_NV_stream_fifo_next"), eglewGetExtension("EGL_NV_stream_fifo_next"));
}

#endif /* EGL_NV_stream_fifo_next */

#ifdef EGL_NV_stream_fifo_synchronous

static void _glewInfo_EGL_NV_stream_fifo_synchronous (void)
{
  glewPrintExt("EGL_NV_stream_fifo_synchronous", EGLEW_NV_stream_fifo_synchronous, eglewIsSupported("EGL_NV_stream_fifo_synchronous"), eglewGetExtension("EGL_NV_stream_fifo_synchronous"));
}

#endif /* EGL_NV_stream_fifo_synchronous */

#ifdef EGL_NV_stream_flush

static void _glewInfo_EGL_NV_stream_flush (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_stream_flush", EGLEW_NV_stream_flush, eglewIsSupported("EGL_NV_stream_flush"), eglewGetExtension("EGL_NV_stream_flush"));

  glewInfoFunc(fi, "eglStreamFlushNV", eglStreamFlushNV == NULL);
}

#endif /* EGL_NV_stream_flush */

#ifdef EGL_NV_stream_frame_limits

static void _glewInfo_EGL_NV_stream_frame_limits (void)
{
  glewPrintExt("EGL_NV_stream_frame_limits", EGLEW_NV_stream_frame_limits, eglewIsSupported("EGL_NV_stream_frame_limits"), eglewGetExtension("EGL_NV_stream_frame_limits"));
}

#endif /* EGL_NV_stream_frame_limits */

#ifdef EGL_NV_stream_metadata

static void _glewInfo_EGL_NV_stream_metadata (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_stream_metadata", EGLEW_NV_stream_metadata, eglewIsSupported("EGL_NV_stream_metadata"), eglewGetExtension("EGL_NV_stream_metadata"));

  glewInfoFunc(fi, "eglQueryDisplayAttribNV", eglQueryDisplayAttribNV == NULL);
  glewInfoFunc(fi, "eglQueryStreamMetadataNV", eglQueryStreamMetadataNV == NULL);
  glewInfoFunc(fi, "eglSetStreamMetadataNV", eglSetStreamMetadataNV == NULL);
}

#endif /* EGL_NV_stream_metadata */

#ifdef EGL_NV_stream_origin

static void _glewInfo_EGL_NV_stream_origin (void)
{
  glewPrintExt("EGL_NV_stream_origin", EGLEW_NV_stream_origin, eglewIsSupported("EGL_NV_stream_origin"), eglewGetExtension("EGL_NV_stream_origin"));
}

#endif /* EGL_NV_stream_origin */

#ifdef EGL_NV_stream_remote

static void _glewInfo_EGL_NV_stream_remote (void)
{
  glewPrintExt("EGL_NV_stream_remote", EGLEW_NV_stream_remote, eglewIsSupported("EGL_NV_stream_remote"), eglewGetExtension("EGL_NV_stream_remote"));
}

#endif /* EGL_NV_stream_remote */

#ifdef EGL_NV_stream_reset

static void _glewInfo_EGL_NV_stream_reset (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_stream_reset", EGLEW_NV_stream_reset, eglewIsSupported("EGL_NV_stream_reset"), eglewGetExtension("EGL_NV_stream_reset"));

  glewInfoFunc(fi, "eglResetStreamNV", eglResetStreamNV == NULL);
}

#endif /* EGL_NV_stream_reset */

#ifdef EGL_NV_stream_socket

static void _glewInfo_EGL_NV_stream_socket (void)
{
  glewPrintExt("EGL_NV_stream_socket", EGLEW_NV_stream_socket, eglewIsSupported("EGL_NV_stream_socket"), eglewGetExtension("EGL_NV_stream_socket"));
}

#endif /* EGL_NV_stream_socket */

#ifdef EGL_NV_stream_socket_inet

static void _glewInfo_EGL_NV_stream_socket_inet (void)
{
  glewPrintExt("EGL_NV_stream_socket_inet", EGLEW_NV_stream_socket_inet, eglewIsSupported("EGL_NV_stream_socket_inet"), eglewGetExtension("EGL_NV_stream_socket_inet"));
}

#endif /* EGL_NV_stream_socket_inet */

#ifdef EGL_NV_stream_socket_unix

static void _glewInfo_EGL_NV_stream_socket_unix (void)
{
  glewPrintExt("EGL_NV_stream_socket_unix", EGLEW_NV_stream_socket_unix, eglewIsSupported("EGL_NV_stream_socket_unix"), eglewGetExtension("EGL_NV_stream_socket_unix"));
}

#endif /* EGL_NV_stream_socket_unix */

#ifdef EGL_NV_stream_sync

static void _glewInfo_EGL_NV_stream_sync (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_stream_sync", EGLEW_NV_stream_sync, eglewIsSupported("EGL_NV_stream_sync"), eglewGetExtension("EGL_NV_stream_sync"));

  glewInfoFunc(fi, "eglCreateStreamSyncNV", eglCreateStreamSyncNV == NULL);
}

#endif /* EGL_NV_stream_sync */

#ifdef EGL_NV_sync

static void _glewInfo_EGL_NV_sync (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_sync", EGLEW_NV_sync, eglewIsSupported("EGL_NV_sync"), eglewGetExtension("EGL_NV_sync"));

  glewInfoFunc(fi, "eglClientWaitSyncNV", eglClientWaitSyncNV == NULL);
  glewInfoFunc(fi, "eglCreateFenceSyncNV", eglCreateFenceSyncNV == NULL);
  glewInfoFunc(fi, "eglDestroySyncNV", eglDestroySyncNV == NULL);
  glewInfoFunc(fi, "eglFenceNV", eglFenceNV == NULL);
  glewInfoFunc(fi, "eglGetSyncAttribNV", eglGetSyncAttribNV == NULL);
  glewInfoFunc(fi, "eglSignalSyncNV", eglSignalSyncNV == NULL);
}

#endif /* EGL_NV_sync */

#ifdef EGL_NV_system_time

static void _glewInfo_EGL_NV_system_time (void)
{
  GLboolean fi = glewPrintExt("EGL_NV_system_time", EGLEW_NV_system_time, eglewIsSupported("EGL_NV_system_time"), eglewGetExtension("EGL_NV_system_time"));

  glewInfoFunc(fi, "eglGetSystemTimeFrequencyNV", eglGetSystemTimeFrequencyNV == NULL);
  glewInfoFunc(fi, "eglGetSystemTimeNV", eglGetSystemTimeNV == NULL);
}

#endif /* EGL_NV_system_time */

#ifdef EGL_NV_triple_buffer

static void _glewInfo_EGL_NV_triple_buffer (void)
{
  glewPrintExt("EGL_NV_triple_buffer", EGLEW_NV_triple_buffer, eglewIsSupported("EGL_NV_triple_buffer"), eglewGetExtension("EGL_NV_triple_buffer"));
}

#endif /* EGL_NV_triple_buffer */

#ifdef EGL_TIZEN_image_native_buffer

static void _glewInfo_EGL_TIZEN_image_native_buffer (void)
{
  glewPrintExt("EGL_TIZEN_image_native_buffer", EGLEW_TIZEN_image_native_buffer, eglewIsSupported("EGL_TIZEN_image_native_buffer"), eglewGetExtension("EGL_TIZEN_image_native_buffer"));
}

#endif /* EGL_TIZEN_image_native_buffer */

#ifdef EGL_TIZEN_image_native_surface

static void _glewInfo_EGL_TIZEN_image_native_surface (void)
{
  glewPrintExt("EGL_TIZEN_image_native_surface", EGLEW_TIZEN_image_native_surface, eglewIsSupported("EGL_TIZEN_image_native_surface"), eglewGetExtension("EGL_TIZEN_image_native_surface"));
}

#endif /* EGL_TIZEN_image_native_surface */

#ifdef EGL_WL_bind_wayland_display

static void _glewInfo_EGL_WL_bind_wayland_display (void)
{
  GLboolean fi = glewPrintExt("EGL_WL_bind_wayland_display", EGLEW_WL_bind_wayland_display, eglewIsSupported("EGL_WL_bind_wayland_display"), eglewGetExtension("EGL_WL_bind_wayland_display"));

  glewInfoFunc(fi, "eglBindWaylandDisplayWL", eglBindWaylandDisplayWL == NULL);
  glewInfoFunc(fi, "eglQueryWaylandBufferWL", eglQueryWaylandBufferWL == NULL);
  glewInfoFunc(fi, "eglUnbindWaylandDisplayWL", eglUnbindWaylandDisplayWL == NULL);
}

#endif /* EGL_WL_bind_wayland_display */

#ifdef EGL_WL_create_wayland_buffer_from_image

static void _glewInfo_EGL_WL_create_wayland_buffer_from_image (void)
{
  GLboolean fi = glewPrintExt("EGL_WL_create_wayland_buffer_from_image", EGLEW_WL_create_wayland_buffer_from_image, eglewIsSupported("EGL_WL_create_wayland_buffer_from_image"), eglewGetExtension("EGL_WL_create_wayland_buffer_from_image"));

  glewInfoFunc(fi, "eglCreateWaylandBufferFromImageWL", eglCreateWaylandBufferFromImageWL == NULL);
}

#endif /* EGL_WL_create_wayland_buffer_from_image */

#elif _WIN32

#ifdef WGL_3DFX_multisample

static void _glewInfo_WGL_3DFX_multisample (void)
{
  glewPrintExt("WGL_3DFX_multisample", WGLEW_3DFX_multisample, wglewIsSupported("WGL_3DFX_multisample"), wglewGetExtension("WGL_3DFX_multisample"));
}

#endif /* WGL_3DFX_multisample */

#ifdef WGL_3DL_stereo_control

static void _glewInfo_WGL_3DL_stereo_control (void)
{
  GLboolean fi = glewPrintExt("WGL_3DL_stereo_control", WGLEW_3DL_stereo_control, wglewIsSupported("WGL_3DL_stereo_control"), wglewGetExtension("WGL_3DL_stereo_control"));

  glewInfoFunc(fi, "wglSetStereoEmitterState3DL", wglSetStereoEmitterState3DL == NULL);
}

#endif /* WGL_3DL_stereo_control */

#ifdef WGL_AMD_gpu_association

static void _glewInfo_WGL_AMD_gpu_association (void)
{
  GLboolean fi = glewPrintExt("WGL_AMD_gpu_association", WGLEW_AMD_gpu_association, wglewIsSupported("WGL_AMD_gpu_association"), wglewGetExtension("WGL_AMD_gpu_association"));

  glewInfoFunc(fi, "wglBlitContextFramebufferAMD", wglBlitContextFramebufferAMD == NULL);
  glewInfoFunc(fi, "wglCreateAssociatedContextAMD", wglCreateAssociatedContextAMD == NULL);
  glewInfoFunc(fi, "wglCreateAssociatedContextAttribsAMD", wglCreateAssociatedContextAttribsAMD == NULL);
  glewInfoFunc(fi, "wglDeleteAssociatedContextAMD", wglDeleteAssociatedContextAMD == NULL);
  glewInfoFunc(fi, "wglGetContextGPUIDAMD", wglGetContextGPUIDAMD == NULL);
  glewInfoFunc(fi, "wglGetCurrentAssociatedContextAMD", wglGetCurrentAssociatedContextAMD == NULL);
  glewInfoFunc(fi, "wglGetGPUIDsAMD", wglGetGPUIDsAMD == NULL);
  glewInfoFunc(fi, "wglGetGPUInfoAMD", wglGetGPUInfoAMD == NULL);
  glewInfoFunc(fi, "wglMakeAssociatedContextCurrentAMD", wglMakeAssociatedContextCurrentAMD == NULL);
}

#endif /* WGL_AMD_gpu_association */

#ifdef WGL_ARB_buffer_region

static void _glewInfo_WGL_ARB_buffer_region (void)
{
  GLboolean fi = glewPrintExt("WGL_ARB_buffer_region", WGLEW_ARB_buffer_region, wglewIsSupported("WGL_ARB_buffer_region"), wglewGetExtension("WGL_ARB_buffer_region"));

  glewInfoFunc(fi, "wglCreateBufferRegionARB", wglCreateBufferRegionARB == NULL);
  glewInfoFunc(fi, "wglDeleteBufferRegionARB", wglDeleteBufferRegionARB == NULL);
  glewInfoFunc(fi, "wglRestoreBufferRegionARB", wglRestoreBufferRegionARB == NULL);
  glewInfoFunc(fi, "wglSaveBufferRegionARB", wglSaveBufferRegionARB == NULL);
}

#endif /* WGL_ARB_buffer_region */

#ifdef WGL_ARB_context_flush_control

static void _glewInfo_WGL_ARB_context_flush_control (void)
{
  glewPrintExt("WGL_ARB_context_flush_control", WGLEW_ARB_context_flush_control, wglewIsSupported("WGL_ARB_context_flush_control"), wglewGetExtension("WGL_ARB_context_flush_control"));
}

#endif /* WGL_ARB_context_flush_control */

#ifdef WGL_ARB_create_context

static void _glewInfo_WGL_ARB_create_context (void)
{
  GLboolean fi = glewPrintExt("WGL_ARB_create_context", WGLEW_ARB_create_context, wglewIsSupported("WGL_ARB_create_context"), wglewGetExtension("WGL_ARB_create_context"));

  glewInfoFunc(fi, "wglCreateContextAttribsARB", wglCreateContextAttribsARB == NULL);
}

#endif /* WGL_ARB_create_context */

#ifdef WGL_ARB_create_context_no_error

static void _glewInfo_WGL_ARB_create_context_no_error (void)
{
  glewPrintExt("WGL_ARB_create_context_no_error", WGLEW_ARB_create_context_no_error, wglewIsSupported("WGL_ARB_create_context_no_error"), wglewGetExtension("WGL_ARB_create_context_no_error"));
}

#endif /* WGL_ARB_create_context_no_error */

#ifdef WGL_ARB_create_context_profile

static void _glewInfo_WGL_ARB_create_context_profile (void)
{
  glewPrintExt("WGL_ARB_create_context_profile", WGLEW_ARB_create_context_profile, wglewIsSupported("WGL_ARB_create_context_profile"), wglewGetExtension("WGL_ARB_create_context_profile"));
}

#endif /* WGL_ARB_create_context_profile */

#ifdef WGL_ARB_create_context_robustness

static void _glewInfo_WGL_ARB_create_context_robustness (void)
{
  glewPrintExt("WGL_ARB_create_context_robustness", WGLEW_ARB_create_context_robustness, wglewIsSupported("WGL_ARB_create_context_robustness"), wglewGetExtension("WGL_ARB_create_context_robustness"));
}

#endif /* WGL_ARB_create_context_robustness */

#ifdef WGL_ARB_extensions_string

static void _glewInfo_WGL_ARB_extensions_string (void)
{
  GLboolean fi = glewPrintExt("WGL_ARB_extensions_string", WGLEW_ARB_extensions_string, wglewIsSupported("WGL_ARB_extensions_string"), wglewGetExtension("WGL_ARB_extensions_string"));

  glewInfoFunc(fi, "wglGetExtensionsStringARB", wglGetExtensionsStringARB == NULL);
}

#endif /* WGL_ARB_extensions_string */

#ifdef WGL_ARB_framebuffer_sRGB

static void _glewInfo_WGL_ARB_framebuffer_sRGB (void)
{
  glewPrintExt("WGL_ARB_framebuffer_sRGB", WGLEW_ARB_framebuffer_sRGB, wglewIsSupported("WGL_ARB_framebuffer_sRGB"), wglewGetExtension("WGL_ARB_framebuffer_sRGB"));
}

#endif /* WGL_ARB_framebuffer_sRGB */

#ifdef WGL_ARB_make_current_read

static void _glewInfo_WGL_ARB_make_current_read (void)
{
  GLboolean fi = glewPrintExt("WGL_ARB_make_current_read", WGLEW_ARB_make_current_read, wglewIsSupported("WGL_ARB_make_current_read"), wglewGetExtension("WGL_ARB_make_current_read"));

  glewInfoFunc(fi, "wglGetCurrentReadDCARB", wglGetCurrentReadDCARB == NULL);
  glewInfoFunc(fi, "wglMakeContextCurrentARB", wglMakeContextCurrentARB == NULL);
}

#endif /* WGL_ARB_make_current_read */

#ifdef WGL_ARB_multisample

static void _glewInfo_WGL_ARB_multisample (void)
{
  glewPrintExt("WGL_ARB_multisample", WGLEW_ARB_multisample, wglewIsSupported("WGL_ARB_multisample"), wglewGetExtension("WGL_ARB_multisample"));
}

#endif /* WGL_ARB_multisample */

#ifdef WGL_ARB_pbuffer

static void _glewInfo_WGL_ARB_pbuffer (void)
{
  GLboolean fi = glewPrintExt("WGL_ARB_pbuffer", WGLEW_ARB_pbuffer, wglewIsSupported("WGL_ARB_pbuffer"), wglewGetExtension("WGL_ARB_pbuffer"));

  glewInfoFunc(fi, "wglCreatePbufferARB", wglCreatePbufferARB == NULL);
  glewInfoFunc(fi, "wglDestroyPbufferARB", wglDestroyPbufferARB == NULL);
  glewInfoFunc(fi, "wglGetPbufferDCARB", wglGetPbufferDCARB == NULL);
  glewInfoFunc(fi, "wglQueryPbufferARB", wglQueryPbufferARB == NULL);
  glewInfoFunc(fi, "wglReleasePbufferDCARB", wglReleasePbufferDCARB == NULL);
}

#endif /* WGL_ARB_pbuffer */

#ifdef WGL_ARB_pixel_format

static void _glewInfo_WGL_ARB_pixel_format (void)
{
  GLboolean fi = glewPrintExt("WGL_ARB_pixel_format", WGLEW_ARB_pixel_format, wglewIsSupported("WGL_ARB_pixel_format"), wglewGetExtension("WGL_ARB_pixel_format"));

  glewInfoFunc(fi, "wglChoosePixelFormatARB", wglChoosePixelFormatARB == NULL);
  glewInfoFunc(fi, "wglGetPixelFormatAttribfvARB", wglGetPixelFormatAttribfvARB == NULL);
  glewInfoFunc(fi, "wglGetPixelFormatAttribivARB", wglGetPixelFormatAttribivARB == NULL);
}

#endif /* WGL_ARB_pixel_format */

#ifdef WGL_ARB_pixel_format_float

static void _glewInfo_WGL_ARB_pixel_format_float (void)
{
  glewPrintExt("WGL_ARB_pixel_format_float", WGLEW_ARB_pixel_format_float, wglewIsSupported("WGL_ARB_pixel_format_float"), wglewGetExtension("WGL_ARB_pixel_format_float"));
}

#endif /* WGL_ARB_pixel_format_float */

#ifdef WGL_ARB_render_texture

static void _glewInfo_WGL_ARB_render_texture (void)
{
  GLboolean fi = glewPrintExt("WGL_ARB_render_texture", WGLEW_ARB_render_texture, wglewIsSupported("WGL_ARB_render_texture"), wglewGetExtension("WGL_ARB_render_texture"));

  glewInfoFunc(fi, "wglBindTexImageARB", wglBindTexImageARB == NULL);
  glewInfoFunc(fi, "wglReleaseTexImageARB", wglReleaseTexImageARB == NULL);
  glewInfoFunc(fi, "wglSetPbufferAttribARB", wglSetPbufferAttribARB == NULL);
}

#endif /* WGL_ARB_render_texture */

#ifdef WGL_ARB_robustness_application_isolation

static void _glewInfo_WGL_ARB_robustness_application_isolation (void)
{
  glewPrintExt("WGL_ARB_robustness_application_isolation", WGLEW_ARB_robustness_application_isolation, wglewIsSupported("WGL_ARB_robustness_application_isolation"), wglewGetExtension("WGL_ARB_robustness_application_isolation"));
}

#endif /* WGL_ARB_robustness_application_isolation */

#ifdef WGL_ARB_robustness_share_group_isolation

static void _glewInfo_WGL_ARB_robustness_share_group_isolation (void)
{
  glewPrintExt("WGL_ARB_robustness_share_group_isolation", WGLEW_ARB_robustness_share_group_isolation, wglewIsSupported("WGL_ARB_robustness_share_group_isolation"), wglewGetExtension("WGL_ARB_robustness_share_group_isolation"));
}

#endif /* WGL_ARB_robustness_share_group_isolation */

#ifdef WGL_ATI_pixel_format_float

static void _glewInfo_WGL_ATI_pixel_format_float (void)
{
  glewPrintExt("WGL_ATI_pixel_format_float", WGLEW_ATI_pixel_format_float, wglewIsSupported("WGL_ATI_pixel_format_float"), wglewGetExtension("WGL_ATI_pixel_format_float"));
}

#endif /* WGL_ATI_pixel_format_float */

#ifdef WGL_ATI_render_texture_rectangle

static void _glewInfo_WGL_ATI_render_texture_rectangle (void)
{
  glewPrintExt("WGL_ATI_render_texture_rectangle", WGLEW_ATI_render_texture_rectangle, wglewIsSupported("WGL_ATI_render_texture_rectangle"), wglewGetExtension("WGL_ATI_render_texture_rectangle"));
}

#endif /* WGL_ATI_render_texture_rectangle */

#ifdef WGL_EXT_colorspace

static void _glewInfo_WGL_EXT_colorspace (void)
{
  glewPrintExt("WGL_EXT_colorspace", WGLEW_EXT_colorspace, wglewIsSupported("WGL_EXT_colorspace"), wglewGetExtension("WGL_EXT_colorspace"));
}

#endif /* WGL_EXT_colorspace */

#ifdef WGL_EXT_create_context_es2_profile

static void _glewInfo_WGL_EXT_create_context_es2_profile (void)
{
  glewPrintExt("WGL_EXT_create_context_es2_profile", WGLEW_EXT_create_context_es2_profile, wglewIsSupported("WGL_EXT_create_context_es2_profile"), wglewGetExtension("WGL_EXT_create_context_es2_profile"));
}

#endif /* WGL_EXT_create_context_es2_profile */

#ifdef WGL_EXT_create_context_es_profile

static void _glewInfo_WGL_EXT_create_context_es_profile (void)
{
  glewPrintExt("WGL_EXT_create_context_es_profile", WGLEW_EXT_create_context_es_profile, wglewIsSupported("WGL_EXT_create_context_es_profile"), wglewGetExtension("WGL_EXT_create_context_es_profile"));
}

#endif /* WGL_EXT_create_context_es_profile */

#ifdef WGL_EXT_depth_float

static void _glewInfo_WGL_EXT_depth_float (void)
{
  glewPrintExt("WGL_EXT_depth_float", WGLEW_EXT_depth_float, wglewIsSupported("WGL_EXT_depth_float"), wglewGetExtension("WGL_EXT_depth_float"));
}

#endif /* WGL_EXT_depth_float */

#ifdef WGL_EXT_display_color_table

static void _glewInfo_WGL_EXT_display_color_table (void)
{
  GLboolean fi = glewPrintExt("WGL_EXT_display_color_table", WGLEW_EXT_display_color_table, wglewIsSupported("WGL_EXT_display_color_table"), wglewGetExtension("WGL_EXT_display_color_table"));

  glewInfoFunc(fi, "wglBindDisplayColorTableEXT", wglBindDisplayColorTableEXT == NULL);
  glewInfoFunc(fi, "wglCreateDisplayColorTableEXT", wglCreateDisplayColorTableEXT == NULL);
  glewInfoFunc(fi, "wglDestroyDisplayColorTableEXT", wglDestroyDisplayColorTableEXT == NULL);
  glewInfoFunc(fi, "wglLoadDisplayColorTableEXT", wglLoadDisplayColorTableEXT == NULL);
}

#endif /* WGL_EXT_display_color_table */

#ifdef WGL_EXT_extensions_string

static void _glewInfo_WGL_EXT_extensions_string (void)
{
  GLboolean fi = glewPrintExt("WGL_EXT_extensions_string", WGLEW_EXT_extensions_string, wglewIsSupported("WGL_EXT_extensions_string"), wglewGetExtension("WGL_EXT_extensions_string"));

  glewInfoFunc(fi, "wglGetExtensionsStringEXT", wglGetExtensionsStringEXT == NULL);
}

#endif /* WGL_EXT_extensions_string */

#ifdef WGL_EXT_framebuffer_sRGB

static void _glewInfo_WGL_EXT_framebuffer_sRGB (void)
{
  glewPrintExt("WGL_EXT_framebuffer_sRGB", WGLEW_EXT_framebuffer_sRGB, wglewIsSupported("WGL_EXT_framebuffer_sRGB"), wglewGetExtension("WGL_EXT_framebuffer_sRGB"));
}

#endif /* WGL_EXT_framebuffer_sRGB */

#ifdef WGL_EXT_make_current_read

static void _glewInfo_WGL_EXT_make_current_read (void)
{
  GLboolean fi = glewPrintExt("WGL_EXT_make_current_read", WGLEW_EXT_make_current_read, wglewIsSupported("WGL_EXT_make_current_read"), wglewGetExtension("WGL_EXT_make_current_read"));

  glewInfoFunc(fi, "wglGetCurrentReadDCEXT", wglGetCurrentReadDCEXT == NULL);
  glewInfoFunc(fi, "wglMakeContextCurrentEXT", wglMakeContextCurrentEXT == NULL);
}

#endif /* WGL_EXT_make_current_read */

#ifdef WGL_EXT_multisample

static void _glewInfo_WGL_EXT_multisample (void)
{
  glewPrintExt("WGL_EXT_multisample", WGLEW_EXT_multisample, wglewIsSupported("WGL_EXT_multisample"), wglewGetExtension("WGL_EXT_multisample"));
}

#endif /* WGL_EXT_multisample */

#ifdef WGL_EXT_pbuffer

static void _glewInfo_WGL_EXT_pbuffer (void)
{
  GLboolean fi = glewPrintExt("WGL_EXT_pbuffer", WGLEW_EXT_pbuffer, wglewIsSupported("WGL_EXT_pbuffer"), wglewGetExtension("WGL_EXT_pbuffer"));

  glewInfoFunc(fi, "wglCreatePbufferEXT", wglCreatePbufferEXT == NULL);
  glewInfoFunc(fi, "wglDestroyPbufferEXT", wglDestroyPbufferEXT == NULL);
  glewInfoFunc(fi, "wglGetPbufferDCEXT", wglGetPbufferDCEXT == NULL);
  glewInfoFunc(fi, "wglQueryPbufferEXT", wglQueryPbufferEXT == NULL);
  glewInfoFunc(fi, "wglReleasePbufferDCEXT", wglReleasePbufferDCEXT == NULL);
}

#endif /* WGL_EXT_pbuffer */

#ifdef WGL_EXT_pixel_format

static void _glewInfo_WGL_EXT_pixel_format (void)
{
  GLboolean fi = glewPrintExt("WGL_EXT_pixel_format", WGLEW_EXT_pixel_format, wglewIsSupported("WGL_EXT_pixel_format"), wglewGetExtension("WGL_EXT_pixel_format"));

  glewInfoFunc(fi, "wglChoosePixelFormatEXT", wglChoosePixelFormatEXT == NULL);
  glewInfoFunc(fi, "wglGetPixelFormatAttribfvEXT", wglGetPixelFormatAttribfvEXT == NULL);
  glewInfoFunc(fi, "wglGetPixelFormatAttribivEXT", wglGetPixelFormatAttribivEXT == NULL);
}

#endif /* WGL_EXT_pixel_format */

#ifdef WGL_EXT_pixel_format_packed_float

static void _glewInfo_WGL_EXT_pixel_format_packed_float (void)
{
  glewPrintExt("WGL_EXT_pixel_format_packed_float", WGLEW_EXT_pixel_format_packed_float, wglewIsSupported("WGL_EXT_pixel_format_packed_float"), wglewGetExtension("WGL_EXT_pixel_format_packed_float"));
}

#endif /* WGL_EXT_pixel_format_packed_float */

#ifdef WGL_EXT_swap_control

static void _glewInfo_WGL_EXT_swap_control (void)
{
  GLboolean fi = glewPrintExt("WGL_EXT_swap_control", WGLEW_EXT_swap_control, wglewIsSupported("WGL_EXT_swap_control"), wglewGetExtension("WGL_EXT_swap_control"));

  glewInfoFunc(fi, "wglGetSwapIntervalEXT", wglGetSwapIntervalEXT == NULL);
  glewInfoFunc(fi, "wglSwapIntervalEXT", wglSwapIntervalEXT == NULL);
}

#endif /* WGL_EXT_swap_control */

#ifdef WGL_EXT_swap_control_tear

static void _glewInfo_WGL_EXT_swap_control_tear (void)
{
  glewPrintExt("WGL_EXT_swap_control_tear", WGLEW_EXT_swap_control_tear, wglewIsSupported("WGL_EXT_swap_control_tear"), wglewGetExtension("WGL_EXT_swap_control_tear"));
}

#endif /* WGL_EXT_swap_control_tear */

#ifdef WGL_I3D_digital_video_control

static void _glewInfo_WGL_I3D_digital_video_control (void)
{
  GLboolean fi = glewPrintExt("WGL_I3D_digital_video_control", WGLEW_I3D_digital_video_control, wglewIsSupported("WGL_I3D_digital_video_control"), wglewGetExtension("WGL_I3D_digital_video_control"));

  glewInfoFunc(fi, "wglGetDigitalVideoParametersI3D", wglGetDigitalVideoParametersI3D == NULL);
  glewInfoFunc(fi, "wglSetDigitalVideoParametersI3D", wglSetDigitalVideoParametersI3D == NULL);
}

#endif /* WGL_I3D_digital_video_control */

#ifdef WGL_I3D_gamma

static void _glewInfo_WGL_I3D_gamma (void)
{
  GLboolean fi = glewPrintExt("WGL_I3D_gamma", WGLEW_I3D_gamma, wglewIsSupported("WGL_I3D_gamma"), wglewGetExtension("WGL_I3D_gamma"));

  glewInfoFunc(fi, "wglGetGammaTableI3D", wglGetGammaTableI3D == NULL);
  glewInfoFunc(fi, "wglGetGammaTableParametersI3D", wglGetGammaTableParametersI3D == NULL);
  glewInfoFunc(fi, "wglSetGammaTableI3D", wglSetGammaTableI3D == NULL);
  glewInfoFunc(fi, "wglSetGammaTableParametersI3D", wglSetGammaTableParametersI3D == NULL);
}

#endif /* WGL_I3D_gamma */

#ifdef WGL_I3D_genlock

static void _glewInfo_WGL_I3D_genlock (void)
{
  GLboolean fi = glewPrintExt("WGL_I3D_genlock", WGLEW_I3D_genlock, wglewIsSupported("WGL_I3D_genlock"), wglewGetExtension("WGL_I3D_genlock"));

  glewInfoFunc(fi, "wglDisableGenlockI3D", wglDisableGenlockI3D == NULL);
  glewInfoFunc(fi, "wglEnableGenlockI3D", wglEnableGenlockI3D == NULL);
  glewInfoFunc(fi, "wglGenlockSampleRateI3D", wglGenlockSampleRateI3D == NULL);
  glewInfoFunc(fi, "wglGenlockSourceDelayI3D", wglGenlockSourceDelayI3D == NULL);
  glewInfoFunc(fi, "wglGenlockSourceEdgeI3D", wglGenlockSourceEdgeI3D == NULL);
  glewInfoFunc(fi, "wglGenlockSourceI3D", wglGenlockSourceI3D == NULL);
  glewInfoFunc(fi, "wglGetGenlockSampleRateI3D", wglGetGenlockSampleRateI3D == NULL);
  glewInfoFunc(fi, "wglGetGenlockSourceDelayI3D", wglGetGenlockSourceDelayI3D == NULL);
  glewInfoFunc(fi, "wglGetGenlockSourceEdgeI3D", wglGetGenlockSourceEdgeI3D == NULL);
  glewInfoFunc(fi, "wglGetGenlockSourceI3D", wglGetGenlockSourceI3D == NULL);
  glewInfoFunc(fi, "wglIsEnabledGenlockI3D", wglIsEnabledGenlockI3D == NULL);
  glewInfoFunc(fi, "wglQueryGenlockMaxSourceDelayI3D", wglQueryGenlockMaxSourceDelayI3D == NULL);
}

#endif /* WGL_I3D_genlock */

#ifdef WGL_I3D_image_buffer

static void _glewInfo_WGL_I3D_image_buffer (void)
{
  GLboolean fi = glewPrintExt("WGL_I3D_image_buffer", WGLEW_I3D_image_buffer, wglewIsSupported("WGL_I3D_image_buffer"), wglewGetExtension("WGL_I3D_image_buffer"));

  glewInfoFunc(fi, "wglAssociateImageBufferEventsI3D", wglAssociateImageBufferEventsI3D == NULL);
  glewInfoFunc(fi, "wglCreateImageBufferI3D", wglCreateImageBufferI3D == NULL);
  glewInfoFunc(fi, "wglDestroyImageBufferI3D", wglDestroyImageBufferI3D == NULL);
  glewInfoFunc(fi, "wglReleaseImageBufferEventsI3D", wglReleaseImageBufferEventsI3D == NULL);
}

#endif /* WGL_I3D_image_buffer */

#ifdef WGL_I3D_swap_frame_lock

static void _glewInfo_WGL_I3D_swap_frame_lock (void)
{
  GLboolean fi = glewPrintExt("WGL_I3D_swap_frame_lock", WGLEW_I3D_swap_frame_lock, wglewIsSupported("WGL_I3D_swap_frame_lock"), wglewGetExtension("WGL_I3D_swap_frame_lock"));

  glewInfoFunc(fi, "wglDisableFrameLockI3D", wglDisableFrameLockI3D == NULL);
  glewInfoFunc(fi, "wglEnableFrameLockI3D", wglEnableFrameLockI3D == NULL);
  glewInfoFunc(fi, "wglIsEnabledFrameLockI3D", wglIsEnabledFrameLockI3D == NULL);
  glewInfoFunc(fi, "wglQueryFrameLockMasterI3D", wglQueryFrameLockMasterI3D == NULL);
}

#endif /* WGL_I3D_swap_frame_lock */

#ifdef WGL_I3D_swap_frame_usage

static void _glewInfo_WGL_I3D_swap_frame_usage (void)
{
  GLboolean fi = glewPrintExt("WGL_I3D_swap_frame_usage", WGLEW_I3D_swap_frame_usage, wglewIsSupported("WGL_I3D_swap_frame_usage"), wglewGetExtension("WGL_I3D_swap_frame_usage"));

  glewInfoFunc(fi, "wglBeginFrameTrackingI3D", wglBeginFrameTrackingI3D == NULL);
  glewInfoFunc(fi, "wglEndFrameTrackingI3D", wglEndFrameTrackingI3D == NULL);
  glewInfoFunc(fi, "wglGetFrameUsageI3D", wglGetFrameUsageI3D == NULL);
  glewInfoFunc(fi, "wglQueryFrameTrackingI3D", wglQueryFrameTrackingI3D == NULL);
}

#endif /* WGL_I3D_swap_frame_usage */

#ifdef WGL_NV_DX_interop

static void _glewInfo_WGL_NV_DX_interop (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_DX_interop", WGLEW_NV_DX_interop, wglewIsSupported("WGL_NV_DX_interop"), wglewGetExtension("WGL_NV_DX_interop"));

  glewInfoFunc(fi, "wglDXCloseDeviceNV", wglDXCloseDeviceNV == NULL);
  glewInfoFunc(fi, "wglDXLockObjectsNV", wglDXLockObjectsNV == NULL);
  glewInfoFunc(fi, "wglDXObjectAccessNV", wglDXObjectAccessNV == NULL);
  glewInfoFunc(fi, "wglDXOpenDeviceNV", wglDXOpenDeviceNV == NULL);
  glewInfoFunc(fi, "wglDXRegisterObjectNV", wglDXRegisterObjectNV == NULL);
  glewInfoFunc(fi, "wglDXSetResourceShareHandleNV", wglDXSetResourceShareHandleNV == NULL);
  glewInfoFunc(fi, "wglDXUnlockObjectsNV", wglDXUnlockObjectsNV == NULL);
  glewInfoFunc(fi, "wglDXUnregisterObjectNV", wglDXUnregisterObjectNV == NULL);
}

#endif /* WGL_NV_DX_interop */

#ifdef WGL_NV_DX_interop2

static void _glewInfo_WGL_NV_DX_interop2 (void)
{
  glewPrintExt("WGL_NV_DX_interop2", WGLEW_NV_DX_interop2, wglewIsSupported("WGL_NV_DX_interop2"), wglewGetExtension("WGL_NV_DX_interop2"));
}

#endif /* WGL_NV_DX_interop2 */

#ifdef WGL_NV_copy_image

static void _glewInfo_WGL_NV_copy_image (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_copy_image", WGLEW_NV_copy_image, wglewIsSupported("WGL_NV_copy_image"), wglewGetExtension("WGL_NV_copy_image"));

  glewInfoFunc(fi, "wglCopyImageSubDataNV", wglCopyImageSubDataNV == NULL);
}

#endif /* WGL_NV_copy_image */

#ifdef WGL_NV_delay_before_swap

static void _glewInfo_WGL_NV_delay_before_swap (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_delay_before_swap", WGLEW_NV_delay_before_swap, wglewIsSupported("WGL_NV_delay_before_swap"), wglewGetExtension("WGL_NV_delay_before_swap"));

  glewInfoFunc(fi, "wglDelayBeforeSwapNV", wglDelayBeforeSwapNV == NULL);
}

#endif /* WGL_NV_delay_before_swap */

#ifdef WGL_NV_float_buffer

static void _glewInfo_WGL_NV_float_buffer (void)
{
  glewPrintExt("WGL_NV_float_buffer", WGLEW_NV_float_buffer, wglewIsSupported("WGL_NV_float_buffer"), wglewGetExtension("WGL_NV_float_buffer"));
}

#endif /* WGL_NV_float_buffer */

#ifdef WGL_NV_gpu_affinity

static void _glewInfo_WGL_NV_gpu_affinity (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_gpu_affinity", WGLEW_NV_gpu_affinity, wglewIsSupported("WGL_NV_gpu_affinity"), wglewGetExtension("WGL_NV_gpu_affinity"));

  glewInfoFunc(fi, "wglCreateAffinityDCNV", wglCreateAffinityDCNV == NULL);
  glewInfoFunc(fi, "wglDeleteDCNV", wglDeleteDCNV == NULL);
  glewInfoFunc(fi, "wglEnumGpuDevicesNV", wglEnumGpuDevicesNV == NULL);
  glewInfoFunc(fi, "wglEnumGpusFromAffinityDCNV", wglEnumGpusFromAffinityDCNV == NULL);
  glewInfoFunc(fi, "wglEnumGpusNV", wglEnumGpusNV == NULL);
}

#endif /* WGL_NV_gpu_affinity */

#ifdef WGL_NV_multigpu_context

static void _glewInfo_WGL_NV_multigpu_context (void)
{
  glewPrintExt("WGL_NV_multigpu_context", WGLEW_NV_multigpu_context, wglewIsSupported("WGL_NV_multigpu_context"), wglewGetExtension("WGL_NV_multigpu_context"));
}

#endif /* WGL_NV_multigpu_context */

#ifdef WGL_NV_multisample_coverage

static void _glewInfo_WGL_NV_multisample_coverage (void)
{
  glewPrintExt("WGL_NV_multisample_coverage", WGLEW_NV_multisample_coverage, wglewIsSupported("WGL_NV_multisample_coverage"), wglewGetExtension("WGL_NV_multisample_coverage"));
}

#endif /* WGL_NV_multisample_coverage */

#ifdef WGL_NV_present_video

static void _glewInfo_WGL_NV_present_video (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_present_video", WGLEW_NV_present_video, wglewIsSupported("WGL_NV_present_video"), wglewGetExtension("WGL_NV_present_video"));

  glewInfoFunc(fi, "wglBindVideoDeviceNV", wglBindVideoDeviceNV == NULL);
  glewInfoFunc(fi, "wglEnumerateVideoDevicesNV", wglEnumerateVideoDevicesNV == NULL);
  glewInfoFunc(fi, "wglQueryCurrentContextNV", wglQueryCurrentContextNV == NULL);
}

#endif /* WGL_NV_present_video */

#ifdef WGL_NV_render_depth_texture

static void _glewInfo_WGL_NV_render_depth_texture (void)
{
  glewPrintExt("WGL_NV_render_depth_texture", WGLEW_NV_render_depth_texture, wglewIsSupported("WGL_NV_render_depth_texture"), wglewGetExtension("WGL_NV_render_depth_texture"));
}

#endif /* WGL_NV_render_depth_texture */

#ifdef WGL_NV_render_texture_rectangle

static void _glewInfo_WGL_NV_render_texture_rectangle (void)
{
  glewPrintExt("WGL_NV_render_texture_rectangle", WGLEW_NV_render_texture_rectangle, wglewIsSupported("WGL_NV_render_texture_rectangle"), wglewGetExtension("WGL_NV_render_texture_rectangle"));
}

#endif /* WGL_NV_render_texture_rectangle */

#ifdef WGL_NV_swap_group

static void _glewInfo_WGL_NV_swap_group (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_swap_group", WGLEW_NV_swap_group, wglewIsSupported("WGL_NV_swap_group"), wglewGetExtension("WGL_NV_swap_group"));

  glewInfoFunc(fi, "wglBindSwapBarrierNV", wglBindSwapBarrierNV == NULL);
  glewInfoFunc(fi, "wglJoinSwapGroupNV", wglJoinSwapGroupNV == NULL);
  glewInfoFunc(fi, "wglQueryFrameCountNV", wglQueryFrameCountNV == NULL);
  glewInfoFunc(fi, "wglQueryMaxSwapGroupsNV", wglQueryMaxSwapGroupsNV == NULL);
  glewInfoFunc(fi, "wglQuerySwapGroupNV", wglQuerySwapGroupNV == NULL);
  glewInfoFunc(fi, "wglResetFrameCountNV", wglResetFrameCountNV == NULL);
}

#endif /* WGL_NV_swap_group */

#ifdef WGL_NV_vertex_array_range

static void _glewInfo_WGL_NV_vertex_array_range (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_vertex_array_range", WGLEW_NV_vertex_array_range, wglewIsSupported("WGL_NV_vertex_array_range"), wglewGetExtension("WGL_NV_vertex_array_range"));

  glewInfoFunc(fi, "wglAllocateMemoryNV", wglAllocateMemoryNV == NULL);
  glewInfoFunc(fi, "wglFreeMemoryNV", wglFreeMemoryNV == NULL);
}

#endif /* WGL_NV_vertex_array_range */

#ifdef WGL_NV_video_capture

static void _glewInfo_WGL_NV_video_capture (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_video_capture", WGLEW_NV_video_capture, wglewIsSupported("WGL_NV_video_capture"), wglewGetExtension("WGL_NV_video_capture"));

  glewInfoFunc(fi, "wglBindVideoCaptureDeviceNV", wglBindVideoCaptureDeviceNV == NULL);
  glewInfoFunc(fi, "wglEnumerateVideoCaptureDevicesNV", wglEnumerateVideoCaptureDevicesNV == NULL);
  glewInfoFunc(fi, "wglLockVideoCaptureDeviceNV", wglLockVideoCaptureDeviceNV == NULL);
  glewInfoFunc(fi, "wglQueryVideoCaptureDeviceNV", wglQueryVideoCaptureDeviceNV == NULL);
  glewInfoFunc(fi, "wglReleaseVideoCaptureDeviceNV", wglReleaseVideoCaptureDeviceNV == NULL);
}

#endif /* WGL_NV_video_capture */

#ifdef WGL_NV_video_output

static void _glewInfo_WGL_NV_video_output (void)
{
  GLboolean fi = glewPrintExt("WGL_NV_video_output", WGLEW_NV_video_output, wglewIsSupported("WGL_NV_video_output"), wglewGetExtension("WGL_NV_video_output"));

  glewInfoFunc(fi, "wglBindVideoImageNV", wglBindVideoImageNV == NULL);
  glewInfoFunc(fi, "wglGetVideoDeviceNV", wglGetVideoDeviceNV == NULL);
  glewInfoFunc(fi, "wglGetVideoInfoNV", wglGetVideoInfoNV == NULL);
  glewInfoFunc(fi, "wglReleaseVideoDeviceNV", wglReleaseVideoDeviceNV == NULL);
  glewInfoFunc(fi, "wglReleaseVideoImageNV", wglReleaseVideoImageNV == NULL);
  glewInfoFunc(fi, "wglSendPbufferToVideoNV", wglSendPbufferToVideoNV == NULL);
}

#endif /* WGL_NV_video_output */

#ifdef WGL_OML_sync_control

static void _glewInfo_WGL_OML_sync_control (void)
{
  GLboolean fi = glewPrintExt("WGL_OML_sync_control", WGLEW_OML_sync_control, wglewIsSupported("WGL_OML_sync_control"), wglewGetExtension("WGL_OML_sync_control"));

  glewInfoFunc(fi, "wglGetMscRateOML", wglGetMscRateOML == NULL);
  glewInfoFunc(fi, "wglGetSyncValuesOML", wglGetSyncValuesOML == NULL);
  glewInfoFunc(fi, "wglSwapBuffersMscOML", wglSwapBuffersMscOML == NULL);
  glewInfoFunc(fi, "wglSwapLayerBuffersMscOML", wglSwapLayerBuffersMscOML == NULL);
  glewInfoFunc(fi, "wglWaitForMscOML", wglWaitForMscOML == NULL);
  glewInfoFunc(fi, "wglWaitForSbcOML", wglWaitForSbcOML == NULL);
}

#endif /* WGL_OML_sync_control */

#else /* _UNIX */

#ifdef GLX_VERSION_1_2

static void _glewInfo_GLX_VERSION_1_2 (void)
{
  GLboolean fi = glewPrintExt("GLX_VERSION_1_2", GLXEW_VERSION_1_2, GLXEW_VERSION_1_2, GLXEW_VERSION_1_2);

  glewInfoFunc(fi, "glXGetCurrentDisplay", glXGetCurrentDisplay == NULL);
}

#endif /* GLX_VERSION_1_2 */

#ifdef GLX_VERSION_1_3

static void _glewInfo_GLX_VERSION_1_3 (void)
{
  GLboolean fi = glewPrintExt("GLX_VERSION_1_3", GLXEW_VERSION_1_3, GLXEW_VERSION_1_3, GLXEW_VERSION_1_3);

  glewInfoFunc(fi, "glXChooseFBConfig", glXChooseFBConfig == NULL);
  glewInfoFunc(fi, "glXCreateNewContext", glXCreateNewContext == NULL);
  glewInfoFunc(fi, "glXCreatePbuffer", glXCreatePbuffer == NULL);
  glewInfoFunc(fi, "glXCreatePixmap", glXCreatePixmap == NULL);
  glewInfoFunc(fi, "glXCreateWindow", glXCreateWindow == NULL);
  glewInfoFunc(fi, "glXDestroyPbuffer", glXDestroyPbuffer == NULL);
  glewInfoFunc(fi, "glXDestroyPixmap", glXDestroyPixmap == NULL);
  glewInfoFunc(fi, "glXDestroyWindow", glXDestroyWindow == NULL);
  glewInfoFunc(fi, "glXGetCurrentReadDrawable", glXGetCurrentReadDrawable == NULL);
  glewInfoFunc(fi, "glXGetFBConfigAttrib", glXGetFBConfigAttrib == NULL);
  glewInfoFunc(fi, "glXGetFBConfigs", glXGetFBConfigs == NULL);
  glewInfoFunc(fi, "glXGetSelectedEvent", glXGetSelectedEvent == NULL);
  glewInfoFunc(fi, "glXGetVisualFromFBConfig", glXGetVisualFromFBConfig == NULL);
  glewInfoFunc(fi, "glXMakeContextCurrent", glXMakeContextCurrent == NULL);
  glewInfoFunc(fi, "glXQueryContext", glXQueryContext == NULL);
  glewInfoFunc(fi, "glXQueryDrawable", glXQueryDrawable == NULL);
  glewInfoFunc(fi, "glXSelectEvent", glXSelectEvent == NULL);
}

#endif /* GLX_VERSION_1_3 */

#ifdef GLX_VERSION_1_4

static void _glewInfo_GLX_VERSION_1_4 (void)
{
  glewPrintExt("GLX_VERSION_1_4", GLXEW_VERSION_1_4, GLXEW_VERSION_1_4, GLXEW_VERSION_1_4);
}

#endif /* GLX_VERSION_1_4 */

#ifdef GLX_3DFX_multisample

static void _glewInfo_GLX_3DFX_multisample (void)
{
  glewPrintExt("GLX_3DFX_multisample", GLXEW_3DFX_multisample, glxewIsSupported("GLX_3DFX_multisample"), glxewGetExtension("GLX_3DFX_multisample"));
}

#endif /* GLX_3DFX_multisample */

#ifdef GLX_AMD_gpu_association

static void _glewInfo_GLX_AMD_gpu_association (void)
{
  GLboolean fi = glewPrintExt("GLX_AMD_gpu_association", GLXEW_AMD_gpu_association, glxewIsSupported("GLX_AMD_gpu_association"), glxewGetExtension("GLX_AMD_gpu_association"));

  glewInfoFunc(fi, "glXBlitContextFramebufferAMD", glXBlitContextFramebufferAMD == NULL);
  glewInfoFunc(fi, "glXCreateAssociatedContextAMD", glXCreateAssociatedContextAMD == NULL);
  glewInfoFunc(fi, "glXCreateAssociatedContextAttribsAMD", glXCreateAssociatedContextAttribsAMD == NULL);
  glewInfoFunc(fi, "glXDeleteAssociatedContextAMD", glXDeleteAssociatedContextAMD == NULL);
  glewInfoFunc(fi, "glXGetContextGPUIDAMD", glXGetContextGPUIDAMD == NULL);
  glewInfoFunc(fi, "glXGetCurrentAssociatedContextAMD", glXGetCurrentAssociatedContextAMD == NULL);
  glewInfoFunc(fi, "glXGetGPUIDsAMD", glXGetGPUIDsAMD == NULL);
  glewInfoFunc(fi, "glXGetGPUInfoAMD", glXGetGPUInfoAMD == NULL);
  glewInfoFunc(fi, "glXMakeAssociatedContextCurrentAMD", glXMakeAssociatedContextCurrentAMD == NULL);
}

#endif /* GLX_AMD_gpu_association */

#ifdef GLX_ARB_context_flush_control

static void _glewInfo_GLX_ARB_context_flush_control (void)
{
  glewPrintExt("GLX_ARB_context_flush_control", GLXEW_ARB_context_flush_control, glxewIsSupported("GLX_ARB_context_flush_control"), glxewGetExtension("GLX_ARB_context_flush_control"));
}

#endif /* GLX_ARB_context_flush_control */

#ifdef GLX_ARB_create_context

static void _glewInfo_GLX_ARB_create_context (void)
{
  GLboolean fi = glewPrintExt("GLX_ARB_create_context", GLXEW_ARB_create_context, glxewIsSupported("GLX_ARB_create_context"), glxewGetExtension("GLX_ARB_create_context"));

  glewInfoFunc(fi, "glXCreateContextAttribsARB", glXCreateContextAttribsARB == NULL);
}

#endif /* GLX_ARB_create_context */

#ifdef GLX_ARB_create_context_no_error

static void _glewInfo_GLX_ARB_create_context_no_error (void)
{
  glewPrintExt("GLX_ARB_create_context_no_error", GLXEW_ARB_create_context_no_error, glxewIsSupported("GLX_ARB_create_context_no_error"), glxewGetExtension("GLX_ARB_create_context_no_error"));
}

#endif /* GLX_ARB_create_context_no_error */

#ifdef GLX_ARB_create_context_profile

static void _glewInfo_GLX_ARB_create_context_profile (void)
{
  glewPrintExt("GLX_ARB_create_context_profile", GLXEW_ARB_create_context_profile, glxewIsSupported("GLX_ARB_create_context_profile"), glxewGetExtension("GLX_ARB_create_context_profile"));
}

#endif /* GLX_ARB_create_context_profile */

#ifdef GLX_ARB_create_context_robustness

static void _glewInfo_GLX_ARB_create_context_robustness (void)
{
  glewPrintExt("GLX_ARB_create_context_robustness", GLXEW_ARB_create_context_robustness, glxewIsSupported("GLX_ARB_create_context_robustness"), glxewGetExtension("GLX_ARB_create_context_robustness"));
}

#endif /* GLX_ARB_create_context_robustness */

#ifdef GLX_ARB_fbconfig_float

static void _glewInfo_GLX_ARB_fbconfig_float (void)
{
  glewPrintExt("GLX_ARB_fbconfig_float", GLXEW_ARB_fbconfig_float, glxewIsSupported("GLX_ARB_fbconfig_float"), glxewGetExtension("GLX_ARB_fbconfig_float"));
}

#endif /* GLX_ARB_fbconfig_float */

#ifdef GLX_ARB_framebuffer_sRGB

static void _glewInfo_GLX_ARB_framebuffer_sRGB (void)
{
  glewPrintExt("GLX_ARB_framebuffer_sRGB", GLXEW_ARB_framebuffer_sRGB, glxewIsSupported("GLX_ARB_framebuffer_sRGB"), glxewGetExtension("GLX_ARB_framebuffer_sRGB"));
}

#endif /* GLX_ARB_framebuffer_sRGB */

#ifdef GLX_ARB_get_proc_address

static void _glewInfo_GLX_ARB_get_proc_address (void)
{
  glewPrintExt("GLX_ARB_get_proc_address", GLXEW_ARB_get_proc_address, glxewIsSupported("GLX_ARB_get_proc_address"), glxewGetExtension("GLX_ARB_get_proc_address"));
}

#endif /* GLX_ARB_get_proc_address */

#ifdef GLX_ARB_multisample

static void _glewInfo_GLX_ARB_multisample (void)
{
  glewPrintExt("GLX_ARB_multisample", GLXEW_ARB_multisample, glxewIsSupported("GLX_ARB_multisample"), glxewGetExtension("GLX_ARB_multisample"));
}

#endif /* GLX_ARB_multisample */

#ifdef GLX_ARB_robustness_application_isolation

static void _glewInfo_GLX_ARB_robustness_application_isolation (void)
{
  glewPrintExt("GLX_ARB_robustness_application_isolation", GLXEW_ARB_robustness_application_isolation, glxewIsSupported("GLX_ARB_robustness_application_isolation"), glxewGetExtension("GLX_ARB_robustness_application_isolation"));
}

#endif /* GLX_ARB_robustness_application_isolation */

#ifdef GLX_ARB_robustness_share_group_isolation

static void _glewInfo_GLX_ARB_robustness_share_group_isolation (void)
{
  glewPrintExt("GLX_ARB_robustness_share_group_isolation", GLXEW_ARB_robustness_share_group_isolation, glxewIsSupported("GLX_ARB_robustness_share_group_isolation"), glxewGetExtension("GLX_ARB_robustness_share_group_isolation"));
}

#endif /* GLX_ARB_robustness_share_group_isolation */

#ifdef GLX_ARB_vertex_buffer_object

static void _glewInfo_GLX_ARB_vertex_buffer_object (void)
{
  glewPrintExt("GLX_ARB_vertex_buffer_object", GLXEW_ARB_vertex_buffer_object, glxewIsSupported("GLX_ARB_vertex_buffer_object"), glxewGetExtension("GLX_ARB_vertex_buffer_object"));
}

#endif /* GLX_ARB_vertex_buffer_object */

#ifdef GLX_ATI_pixel_format_float

static void _glewInfo_GLX_ATI_pixel_format_float (void)
{
  glewPrintExt("GLX_ATI_pixel_format_float", GLXEW_ATI_pixel_format_float, glxewIsSupported("GLX_ATI_pixel_format_float"), glxewGetExtension("GLX_ATI_pixel_format_float"));
}

#endif /* GLX_ATI_pixel_format_float */

#ifdef GLX_ATI_render_texture

static void _glewInfo_GLX_ATI_render_texture (void)
{
  GLboolean fi = glewPrintExt("GLX_ATI_render_texture", GLXEW_ATI_render_texture, glxewIsSupported("GLX_ATI_render_texture"), glxewGetExtension("GLX_ATI_render_texture"));

  glewInfoFunc(fi, "glXBindTexImageATI", glXBindTexImageATI == NULL);
  glewInfoFunc(fi, "glXDrawableAttribATI", glXDrawableAttribATI == NULL);
  glewInfoFunc(fi, "glXReleaseTexImageATI", glXReleaseTexImageATI == NULL);
}

#endif /* GLX_ATI_render_texture */

#ifdef GLX_EXT_buffer_age

static void _glewInfo_GLX_EXT_buffer_age (void)
{
  glewPrintExt("GLX_EXT_buffer_age", GLXEW_EXT_buffer_age, glxewIsSupported("GLX_EXT_buffer_age"), glxewGetExtension("GLX_EXT_buffer_age"));
}

#endif /* GLX_EXT_buffer_age */

#ifdef GLX_EXT_context_priority

static void _glewInfo_GLX_EXT_context_priority (void)
{
  glewPrintExt("GLX_EXT_context_priority", GLXEW_EXT_context_priority, glxewIsSupported("GLX_EXT_context_priority"), glxewGetExtension("GLX_EXT_context_priority"));
}

#endif /* GLX_EXT_context_priority */

#ifdef GLX_EXT_create_context_es2_profile

static void _glewInfo_GLX_EXT_create_context_es2_profile (void)
{
  glewPrintExt("GLX_EXT_create_context_es2_profile", GLXEW_EXT_create_context_es2_profile, glxewIsSupported("GLX_EXT_create_context_es2_profile"), glxewGetExtension("GLX_EXT_create_context_es2_profile"));
}

#endif /* GLX_EXT_create_context_es2_profile */

#ifdef GLX_EXT_create_context_es_profile

static void _glewInfo_GLX_EXT_create_context_es_profile (void)
{
  glewPrintExt("GLX_EXT_create_context_es_profile", GLXEW_EXT_create_context_es_profile, glxewIsSupported("GLX_EXT_create_context_es_profile"), glxewGetExtension("GLX_EXT_create_context_es_profile"));
}

#endif /* GLX_EXT_create_context_es_profile */

#ifdef GLX_EXT_fbconfig_packed_float

static void _glewInfo_GLX_EXT_fbconfig_packed_float (void)
{
  glewPrintExt("GLX_EXT_fbconfig_packed_float", GLXEW_EXT_fbconfig_packed_float, glxewIsSupported("GLX_EXT_fbconfig_packed_float"), glxewGetExtension("GLX_EXT_fbconfig_packed_float"));
}

#endif /* GLX_EXT_fbconfig_packed_float */

#ifdef GLX_EXT_framebuffer_sRGB

static void _glewInfo_GLX_EXT_framebuffer_sRGB (void)
{
  glewPrintExt("GLX_EXT_framebuffer_sRGB", GLXEW_EXT_framebuffer_sRGB, glxewIsSupported("GLX_EXT_framebuffer_sRGB"), glxewGetExtension("GLX_EXT_framebuffer_sRGB"));
}

#endif /* GLX_EXT_framebuffer_sRGB */

#ifdef GLX_EXT_import_context

static void _glewInfo_GLX_EXT_import_context (void)
{
  GLboolean fi = glewPrintExt("GLX_EXT_import_context", GLXEW_EXT_import_context, glxewIsSupported("GLX_EXT_import_context"), glxewGetExtension("GLX_EXT_import_context"));

  glewInfoFunc(fi, "glXFreeContextEXT", glXFreeContextEXT == NULL);
  glewInfoFunc(fi, "glXGetContextIDEXT", glXGetContextIDEXT == NULL);
  glewInfoFunc(fi, "glXGetCurrentDisplayEXT", glXGetCurrentDisplayEXT == NULL);
  glewInfoFunc(fi, "glXImportContextEXT", glXImportContextEXT == NULL);
  glewInfoFunc(fi, "glXQueryContextInfoEXT", glXQueryContextInfoEXT == NULL);
}

#endif /* GLX_EXT_import_context */

#ifdef GLX_EXT_libglvnd

static void _glewInfo_GLX_EXT_libglvnd (void)
{
  glewPrintExt("GLX_EXT_libglvnd", GLXEW_EXT_libglvnd, glxewIsSupported("GLX_EXT_libglvnd"), glxewGetExtension("GLX_EXT_libglvnd"));
}

#endif /* GLX_EXT_libglvnd */

#ifdef GLX_EXT_no_config_context

static void _glewInfo_GLX_EXT_no_config_context (void)
{
  glewPrintExt("GLX_EXT_no_config_context", GLXEW_EXT_no_config_context, glxewIsSupported("GLX_EXT_no_config_context"), glxewGetExtension("GLX_EXT_no_config_context"));
}

#endif /* GLX_EXT_no_config_context */

#ifdef GLX_EXT_scene_marker

static void _glewInfo_GLX_EXT_scene_marker (void)
{
  glewPrintExt("GLX_EXT_scene_marker", GLXEW_EXT_scene_marker, glxewIsSupported("GLX_EXT_scene_marker"), glxewGetExtension("GLX_EXT_scene_marker"));
}

#endif /* GLX_EXT_scene_marker */

#ifdef GLX_EXT_stereo_tree

static void _glewInfo_GLX_EXT_stereo_tree (void)
{
  glewPrintExt("GLX_EXT_stereo_tree", GLXEW_EXT_stereo_tree, glxewIsSupported("GLX_EXT_stereo_tree"), glxewGetExtension("GLX_EXT_stereo_tree"));
}

#endif /* GLX_EXT_stereo_tree */

#ifdef GLX_EXT_swap_control

static void _glewInfo_GLX_EXT_swap_control (void)
{
  GLboolean fi = glewPrintExt("GLX_EXT_swap_control", GLXEW_EXT_swap_control, glxewIsSupported("GLX_EXT_swap_control"), glxewGetExtension("GLX_EXT_swap_control"));

  glewInfoFunc(fi, "glXSwapIntervalEXT", glXSwapIntervalEXT == NULL);
}

#endif /* GLX_EXT_swap_control */

#ifdef GLX_EXT_swap_control_tear

static void _glewInfo_GLX_EXT_swap_control_tear (void)
{
  glewPrintExt("GLX_EXT_swap_control_tear", GLXEW_EXT_swap_control_tear, glxewIsSupported("GLX_EXT_swap_control_tear"), glxewGetExtension("GLX_EXT_swap_control_tear"));
}

#endif /* GLX_EXT_swap_control_tear */

#ifdef GLX_EXT_texture_from_pixmap

static void _glewInfo_GLX_EXT_texture_from_pixmap (void)
{
  GLboolean fi = glewPrintExt("GLX_EXT_texture_from_pixmap", GLXEW_EXT_texture_from_pixmap, glxewIsSupported("GLX_EXT_texture_from_pixmap"), glxewGetExtension("GLX_EXT_texture_from_pixmap"));

  glewInfoFunc(fi, "glXBindTexImageEXT", glXBindTexImageEXT == NULL);
  glewInfoFunc(fi, "glXReleaseTexImageEXT", glXReleaseTexImageEXT == NULL);
}

#endif /* GLX_EXT_texture_from_pixmap */

#ifdef GLX_EXT_visual_info

static void _glewInfo_GLX_EXT_visual_info (void)
{
  glewPrintExt("GLX_EXT_visual_info", GLXEW_EXT_visual_info, glxewIsSupported("GLX_EXT_visual_info"), glxewGetExtension("GLX_EXT_visual_info"));
}

#endif /* GLX_EXT_visual_info */

#ifdef GLX_EXT_visual_rating

static void _glewInfo_GLX_EXT_visual_rating (void)
{
  glewPrintExt("GLX_EXT_visual_rating", GLXEW_EXT_visual_rating, glxewIsSupported("GLX_EXT_visual_rating"), glxewGetExtension("GLX_EXT_visual_rating"));
}

#endif /* GLX_EXT_visual_rating */

#ifdef GLX_INTEL_swap_event

static void _glewInfo_GLX_INTEL_swap_event (void)
{
  glewPrintExt("GLX_INTEL_swap_event", GLXEW_INTEL_swap_event, glxewIsSupported("GLX_INTEL_swap_event"), glxewGetExtension("GLX_INTEL_swap_event"));
}

#endif /* GLX_INTEL_swap_event */

#ifdef GLX_MESA_agp_offset

static void _glewInfo_GLX_MESA_agp_offset (void)
{
  GLboolean fi = glewPrintExt("GLX_MESA_agp_offset", GLXEW_MESA_agp_offset, glxewIsSupported("GLX_MESA_agp_offset"), glxewGetExtension("GLX_MESA_agp_offset"));

  glewInfoFunc(fi, "glXGetAGPOffsetMESA", glXGetAGPOffsetMESA == NULL);
}

#endif /* GLX_MESA_agp_offset */

#ifdef GLX_MESA_copy_sub_buffer

static void _glewInfo_GLX_MESA_copy_sub_buffer (void)
{
  GLboolean fi = glewPrintExt("GLX_MESA_copy_sub_buffer", GLXEW_MESA_copy_sub_buffer, glxewIsSupported("GLX_MESA_copy_sub_buffer"), glxewGetExtension("GLX_MESA_copy_sub_buffer"));

  glewInfoFunc(fi, "glXCopySubBufferMESA", glXCopySubBufferMESA == NULL);
}

#endif /* GLX_MESA_copy_sub_buffer */

#ifdef GLX_MESA_pixmap_colormap

static void _glewInfo_GLX_MESA_pixmap_colormap (void)
{
  GLboolean fi = glewPrintExt("GLX_MESA_pixmap_colormap", GLXEW_MESA_pixmap_colormap, glxewIsSupported("GLX_MESA_pixmap_colormap"), glxewGetExtension("GLX_MESA_pixmap_colormap"));

  glewInfoFunc(fi, "glXCreateGLXPixmapMESA", glXCreateGLXPixmapMESA == NULL);
}

#endif /* GLX_MESA_pixmap_colormap */

#ifdef GLX_MESA_query_renderer

static void _glewInfo_GLX_MESA_query_renderer (void)
{
  GLboolean fi = glewPrintExt("GLX_MESA_query_renderer", GLXEW_MESA_query_renderer, glxewIsSupported("GLX_MESA_query_renderer"), glxewGetExtension("GLX_MESA_query_renderer"));

  glewInfoFunc(fi, "glXQueryCurrentRendererIntegerMESA", glXQueryCurrentRendererIntegerMESA == NULL);
  glewInfoFunc(fi, "glXQueryCurrentRendererStringMESA", glXQueryCurrentRendererStringMESA == NULL);
  glewInfoFunc(fi, "glXQueryRendererIntegerMESA", glXQueryRendererIntegerMESA == NULL);
  glewInfoFunc(fi, "glXQueryRendererStringMESA", glXQueryRendererStringMESA == NULL);
}

#endif /* GLX_MESA_query_renderer */

#ifdef GLX_MESA_release_buffers

static void _glewInfo_GLX_MESA_release_buffers (void)
{
  GLboolean fi = glewPrintExt("GLX_MESA_release_buffers", GLXEW_MESA_release_buffers, glxewIsSupported("GLX_MESA_release_buffers"), glxewGetExtension("GLX_MESA_release_buffers"));

  glewInfoFunc(fi, "glXReleaseBuffersMESA", glXReleaseBuffersMESA == NULL);
}

#endif /* GLX_MESA_release_buffers */

#ifdef GLX_MESA_set_3dfx_mode

static void _glewInfo_GLX_MESA_set_3dfx_mode (void)
{
  GLboolean fi = glewPrintExt("GLX_MESA_set_3dfx_mode", GLXEW_MESA_set_3dfx_mode, glxewIsSupported("GLX_MESA_set_3dfx_mode"), glxewGetExtension("GLX_MESA_set_3dfx_mode"));

  glewInfoFunc(fi, "glXSet3DfxModeMESA", glXSet3DfxModeMESA == NULL);
}

#endif /* GLX_MESA_set_3dfx_mode */

#ifdef GLX_MESA_swap_control

static void _glewInfo_GLX_MESA_swap_control (void)
{
  GLboolean fi = glewPrintExt("GLX_MESA_swap_control", GLXEW_MESA_swap_control, glxewIsSupported("GLX_MESA_swap_control"), glxewGetExtension("GLX_MESA_swap_control"));

  glewInfoFunc(fi, "glXGetSwapIntervalMESA", glXGetSwapIntervalMESA == NULL);
  glewInfoFunc(fi, "glXSwapIntervalMESA", glXSwapIntervalMESA == NULL);
}

#endif /* GLX_MESA_swap_control */

#ifdef GLX_NV_copy_buffer

static void _glewInfo_GLX_NV_copy_buffer (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_copy_buffer", GLXEW_NV_copy_buffer, glxewIsSupported("GLX_NV_copy_buffer"), glxewGetExtension("GLX_NV_copy_buffer"));

  glewInfoFunc(fi, "glXCopyBufferSubDataNV", glXCopyBufferSubDataNV == NULL);
  glewInfoFunc(fi, "glXNamedCopyBufferSubDataNV", glXNamedCopyBufferSubDataNV == NULL);
}

#endif /* GLX_NV_copy_buffer */

#ifdef GLX_NV_copy_image

static void _glewInfo_GLX_NV_copy_image (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_copy_image", GLXEW_NV_copy_image, glxewIsSupported("GLX_NV_copy_image"), glxewGetExtension("GLX_NV_copy_image"));

  glewInfoFunc(fi, "glXCopyImageSubDataNV", glXCopyImageSubDataNV == NULL);
}

#endif /* GLX_NV_copy_image */

#ifdef GLX_NV_delay_before_swap

static void _glewInfo_GLX_NV_delay_before_swap (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_delay_before_swap", GLXEW_NV_delay_before_swap, glxewIsSupported("GLX_NV_delay_before_swap"), glxewGetExtension("GLX_NV_delay_before_swap"));

  glewInfoFunc(fi, "glXDelayBeforeSwapNV", glXDelayBeforeSwapNV == NULL);
}

#endif /* GLX_NV_delay_before_swap */

#ifdef GLX_NV_float_buffer

static void _glewInfo_GLX_NV_float_buffer (void)
{
  glewPrintExt("GLX_NV_float_buffer", GLXEW_NV_float_buffer, glxewIsSupported("GLX_NV_float_buffer"), glxewGetExtension("GLX_NV_float_buffer"));
}

#endif /* GLX_NV_float_buffer */

#ifdef GLX_NV_multigpu_context

static void _glewInfo_GLX_NV_multigpu_context (void)
{
  glewPrintExt("GLX_NV_multigpu_context", GLXEW_NV_multigpu_context, glxewIsSupported("GLX_NV_multigpu_context"), glxewGetExtension("GLX_NV_multigpu_context"));
}

#endif /* GLX_NV_multigpu_context */

#ifdef GLX_NV_multisample_coverage

static void _glewInfo_GLX_NV_multisample_coverage (void)
{
  glewPrintExt("GLX_NV_multisample_coverage", GLXEW_NV_multisample_coverage, glxewIsSupported("GLX_NV_multisample_coverage"), glxewGetExtension("GLX_NV_multisample_coverage"));
}

#endif /* GLX_NV_multisample_coverage */

#ifdef GLX_NV_present_video

static void _glewInfo_GLX_NV_present_video (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_present_video", GLXEW_NV_present_video, glxewIsSupported("GLX_NV_present_video"), glxewGetExtension("GLX_NV_present_video"));

  glewInfoFunc(fi, "glXBindVideoDeviceNV", glXBindVideoDeviceNV == NULL);
  glewInfoFunc(fi, "glXEnumerateVideoDevicesNV", glXEnumerateVideoDevicesNV == NULL);
}

#endif /* GLX_NV_present_video */

#ifdef GLX_NV_robustness_video_memory_purge

static void _glewInfo_GLX_NV_robustness_video_memory_purge (void)
{
  glewPrintExt("GLX_NV_robustness_video_memory_purge", GLXEW_NV_robustness_video_memory_purge, glxewIsSupported("GLX_NV_robustness_video_memory_purge"), glxewGetExtension("GLX_NV_robustness_video_memory_purge"));
}

#endif /* GLX_NV_robustness_video_memory_purge */

#ifdef GLX_NV_swap_group

static void _glewInfo_GLX_NV_swap_group (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_swap_group", GLXEW_NV_swap_group, glxewIsSupported("GLX_NV_swap_group"), glxewGetExtension("GLX_NV_swap_group"));

  glewInfoFunc(fi, "glXBindSwapBarrierNV", glXBindSwapBarrierNV == NULL);
  glewInfoFunc(fi, "glXJoinSwapGroupNV", glXJoinSwapGroupNV == NULL);
  glewInfoFunc(fi, "glXQueryFrameCountNV", glXQueryFrameCountNV == NULL);
  glewInfoFunc(fi, "glXQueryMaxSwapGroupsNV", glXQueryMaxSwapGroupsNV == NULL);
  glewInfoFunc(fi, "glXQuerySwapGroupNV", glXQuerySwapGroupNV == NULL);
  glewInfoFunc(fi, "glXResetFrameCountNV", glXResetFrameCountNV == NULL);
}

#endif /* GLX_NV_swap_group */

#ifdef GLX_NV_vertex_array_range

static void _glewInfo_GLX_NV_vertex_array_range (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_vertex_array_range", GLXEW_NV_vertex_array_range, glxewIsSupported("GLX_NV_vertex_array_range"), glxewGetExtension("GLX_NV_vertex_array_range"));

  glewInfoFunc(fi, "glXAllocateMemoryNV", glXAllocateMemoryNV == NULL);
  glewInfoFunc(fi, "glXFreeMemoryNV", glXFreeMemoryNV == NULL);
}

#endif /* GLX_NV_vertex_array_range */

#ifdef GLX_NV_video_capture

static void _glewInfo_GLX_NV_video_capture (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_video_capture", GLXEW_NV_video_capture, glxewIsSupported("GLX_NV_video_capture"), glxewGetExtension("GLX_NV_video_capture"));

  glewInfoFunc(fi, "glXBindVideoCaptureDeviceNV", glXBindVideoCaptureDeviceNV == NULL);
  glewInfoFunc(fi, "glXEnumerateVideoCaptureDevicesNV", glXEnumerateVideoCaptureDevicesNV == NULL);
  glewInfoFunc(fi, "glXLockVideoCaptureDeviceNV", glXLockVideoCaptureDeviceNV == NULL);
  glewInfoFunc(fi, "glXQueryVideoCaptureDeviceNV", glXQueryVideoCaptureDeviceNV == NULL);
  glewInfoFunc(fi, "glXReleaseVideoCaptureDeviceNV", glXReleaseVideoCaptureDeviceNV == NULL);
}

#endif /* GLX_NV_video_capture */

#ifdef GLX_NV_video_out

static void _glewInfo_GLX_NV_video_out (void)
{
  GLboolean fi = glewPrintExt("GLX_NV_video_out", GLXEW_NV_video_out, glxewIsSupported("GLX_NV_video_out"), glxewGetExtension("GLX_NV_video_out"));

  glewInfoFunc(fi, "glXBindVideoImageNV", glXBindVideoImageNV == NULL);
  glewInfoFunc(fi, "glXGetVideoDeviceNV", glXGetVideoDeviceNV == NULL);
  glewInfoFunc(fi, "glXGetVideoInfoNV", glXGetVideoInfoNV == NULL);
  glewInfoFunc(fi, "glXReleaseVideoDeviceNV", glXReleaseVideoDeviceNV == NULL);
  glewInfoFunc(fi, "glXReleaseVideoImageNV", glXReleaseVideoImageNV == NULL);
  glewInfoFunc(fi, "glXSendPbufferToVideoNV", glXSendPbufferToVideoNV == NULL);
}

#endif /* GLX_NV_video_out */

#ifdef GLX_OML_swap_method

static void _glewInfo_GLX_OML_swap_method (void)
{
  glewPrintExt("GLX_OML_swap_method", GLXEW_OML_swap_method, glxewIsSupported("GLX_OML_swap_method"), glxewGetExtension("GLX_OML_swap_method"));
}

#endif /* GLX_OML_swap_method */

#ifdef GLX_OML_sync_control

static void _glewInfo_GLX_OML_sync_control (void)
{
  GLboolean fi = glewPrintExt("GLX_OML_sync_control", GLXEW_OML_sync_control, glxewIsSupported("GLX_OML_sync_control"), glxewGetExtension("GLX_OML_sync_control"));

  glewInfoFunc(fi, "glXGetMscRateOML", glXGetMscRateOML == NULL);
  glewInfoFunc(fi, "glXGetSyncValuesOML", glXGetSyncValuesOML == NULL);
  glewInfoFunc(fi, "glXSwapBuffersMscOML", glXSwapBuffersMscOML == NULL);
  glewInfoFunc(fi, "glXWaitForMscOML", glXWaitForMscOML == NULL);
  glewInfoFunc(fi, "glXWaitForSbcOML", glXWaitForSbcOML == NULL);
}

#endif /* GLX_OML_sync_control */

#ifdef GLX_SGIS_blended_overlay

static void _glewInfo_GLX_SGIS_blended_overlay (void)
{
  glewPrintExt("GLX_SGIS_blended_overlay", GLXEW_SGIS_blended_overlay, glxewIsSupported("GLX_SGIS_blended_overlay"), glxewGetExtension("GLX_SGIS_blended_overlay"));
}

#endif /* GLX_SGIS_blended_overlay */

#ifdef GLX_SGIS_color_range

static void _glewInfo_GLX_SGIS_color_range (void)
{
  glewPrintExt("GLX_SGIS_color_range", GLXEW_SGIS_color_range, glxewIsSupported("GLX_SGIS_color_range"), glxewGetExtension("GLX_SGIS_color_range"));
}

#endif /* GLX_SGIS_color_range */

#ifdef GLX_SGIS_multisample

static void _glewInfo_GLX_SGIS_multisample (void)
{
  glewPrintExt("GLX_SGIS_multisample", GLXEW_SGIS_multisample, glxewIsSupported("GLX_SGIS_multisample"), glxewGetExtension("GLX_SGIS_multisample"));
}

#endif /* GLX_SGIS_multisample */

#ifdef GLX_SGIS_shared_multisample

static void _glewInfo_GLX_SGIS_shared_multisample (void)
{
  glewPrintExt("GLX_SGIS_shared_multisample", GLXEW_SGIS_shared_multisample, glxewIsSupported("GLX_SGIS_shared_multisample"), glxewGetExtension("GLX_SGIS_shared_multisample"));
}

#endif /* GLX_SGIS_shared_multisample */

#ifdef GLX_SGIX_fbconfig

static void _glewInfo_GLX_SGIX_fbconfig (void)
{
  GLboolean fi = glewPrintExt("GLX_SGIX_fbconfig", GLXEW_SGIX_fbconfig, glxewIsSupported("GLX_SGIX_fbconfig"), glxewGetExtension("GLX_SGIX_fbconfig"));

  glewInfoFunc(fi, "glXChooseFBConfigSGIX", glXChooseFBConfigSGIX == NULL);
  glewInfoFunc(fi, "glXCreateContextWithConfigSGIX", glXCreateContextWithConfigSGIX == NULL);
  glewInfoFunc(fi, "glXCreateGLXPixmapWithConfigSGIX", glXCreateGLXPixmapWithConfigSGIX == NULL);
  glewInfoFunc(fi, "glXGetFBConfigAttribSGIX", glXGetFBConfigAttribSGIX == NULL);
  glewInfoFunc(fi, "glXGetFBConfigFromVisualSGIX", glXGetFBConfigFromVisualSGIX == NULL);
  glewInfoFunc(fi, "glXGetVisualFromFBConfigSGIX", glXGetVisualFromFBConfigSGIX == NULL);
}

#endif /* GLX_SGIX_fbconfig */

#ifdef GLX_SGIX_hyperpipe

static void _glewInfo_GLX_SGIX_hyperpipe (void)
{
  GLboolean fi = glewPrintExt("GLX_SGIX_hyperpipe", GLXEW_SGIX_hyperpipe, glxewIsSupported("GLX_SGIX_hyperpipe"), glxewGetExtension("GLX_SGIX_hyperpipe"));

  glewInfoFunc(fi, "glXBindHyperpipeSGIX", glXBindHyperpipeSGIX == NULL);
  glewInfoFunc(fi, "glXDestroyHyperpipeConfigSGIX", glXDestroyHyperpipeConfigSGIX == NULL);
  glewInfoFunc(fi, "glXHyperpipeAttribSGIX", glXHyperpipeAttribSGIX == NULL);
  glewInfoFunc(fi, "glXHyperpipeConfigSGIX", glXHyperpipeConfigSGIX == NULL);
  glewInfoFunc(fi, "glXQueryHyperpipeAttribSGIX", glXQueryHyperpipeAttribSGIX == NULL);
  glewInfoFunc(fi, "glXQueryHyperpipeBestAttribSGIX", glXQueryHyperpipeBestAttribSGIX == NULL);
  glewInfoFunc(fi, "glXQueryHyperpipeConfigSGIX", glXQueryHyperpipeConfigSGIX == NULL);
  glewInfoFunc(fi, "glXQueryHyperpipeNetworkSGIX", glXQueryHyperpipeNetworkSGIX == NULL);
}

#endif /* GLX_SGIX_hyperpipe */

#ifdef GLX_SGIX_pbuffer

static void _glewInfo_GLX_SGIX_pbuffer (void)
{
  GLboolean fi = glewPrintExt("GLX_SGIX_pbuffer", GLXEW_SGIX_pbuffer, glxewIsSupported("GLX_SGIX_pbuffer"), glxewGetExtension("GLX_SGIX_pbuffer"));

  glewInfoFunc(fi, "glXCreateGLXPbufferSGIX", glXCreateGLXPbufferSGIX == NULL);
  glewInfoFunc(fi, "glXDestroyGLXPbufferSGIX", glXDestroyGLXPbufferSGIX == NULL);
  glewInfoFunc(fi, "glXGetSelectedEventSGIX", glXGetSelectedEventSGIX == NULL);
  glewInfoFunc(fi, "glXQueryGLXPbufferSGIX", glXQueryGLXPbufferSGIX == NULL);
  glewInfoFunc(fi, "glXSelectEventSGIX", glXSelectEventSGIX == NULL);
}

#endif /* GLX_SGIX_pbuffer */

#ifdef GLX_SGIX_swap_barrier

static void _glewInfo_GLX_SGIX_swap_barrier (void)
{
  GLboolean fi = glewPrintExt("GLX_SGIX_swap_barrier", GLXEW_SGIX_swap_barrier, glxewIsSupported("GLX_SGIX_swap_barrier"), glxewGetExtension("GLX_SGIX_swap_barrier"));

  glewInfoFunc(fi, "glXBindSwapBarrierSGIX", glXBindSwapBarrierSGIX == NULL);
  glewInfoFunc(fi, "glXQueryMaxSwapBarriersSGIX", glXQueryMaxSwapBarriersSGIX == NULL);
}

#endif /* GLX_SGIX_swap_barrier */

#ifdef GLX_SGIX_swap_group

static void _glewInfo_GLX_SGIX_swap_group (void)
{
  GLboolean fi = glewPrintExt("GLX_SGIX_swap_group", GLXEW_SGIX_swap_group, glxewIsSupported("GLX_SGIX_swap_group"), glxewGetExtension("GLX_SGIX_swap_group"));

  glewInfoFunc(fi, "glXJoinSwapGroupSGIX", glXJoinSwapGroupSGIX == NULL);
}

#endif /* GLX_SGIX_swap_group */

#ifdef GLX_SGIX_video_resize

static void _glewInfo_GLX_SGIX_video_resize (void)
{
  GLboolean fi = glewPrintExt("GLX_SGIX_video_resize", GLXEW_SGIX_video_resize, glxewIsSupported("GLX_SGIX_video_resize"), glxewGetExtension("GLX_SGIX_video_resize"));

  glewInfoFunc(fi, "glXBindChannelToWindowSGIX", glXBindChannelToWindowSGIX == NULL);
  glewInfoFunc(fi, "glXChannelRectSGIX", glXChannelRectSGIX == NULL);
  glewInfoFunc(fi, "glXChannelRectSyncSGIX", glXChannelRectSyncSGIX == NULL);
  glewInfoFunc(fi, "glXQueryChannelDeltasSGIX", glXQueryChannelDeltasSGIX == NULL);
  glewInfoFunc(fi, "glXQueryChannelRectSGIX", glXQueryChannelRectSGIX == NULL);
}

#endif /* GLX_SGIX_video_resize */

#ifdef GLX_SGIX_visual_select_group

static void _glewInfo_GLX_SGIX_visual_select_group (void)
{
  glewPrintExt("GLX_SGIX_visual_select_group", GLXEW_SGIX_visual_select_group, glxewIsSupported("GLX_SGIX_visual_select_group"), glxewGetExtension("GLX_SGIX_visual_select_group"));
}

#endif /* GLX_SGIX_visual_select_group */

#ifdef GLX_SGI_cushion

static void _glewInfo_GLX_SGI_cushion (void)
{
  GLboolean fi = glewPrintExt("GLX_SGI_cushion", GLXEW_SGI_cushion, glxewIsSupported("GLX_SGI_cushion"), glxewGetExtension("GLX_SGI_cushion"));

  glewInfoFunc(fi, "glXCushionSGI", glXCushionSGI == NULL);
}

#endif /* GLX_SGI_cushion */

#ifdef GLX_SGI_make_current_read

static void _glewInfo_GLX_SGI_make_current_read (void)
{
  GLboolean fi = glewPrintExt("GLX_SGI_make_current_read", GLXEW_SGI_make_current_read, glxewIsSupported("GLX_SGI_make_current_read"), glxewGetExtension("GLX_SGI_make_current_read"));

  glewInfoFunc(fi, "glXGetCurrentReadDrawableSGI", glXGetCurrentReadDrawableSGI == NULL);
  glewInfoFunc(fi, "glXMakeCurrentReadSGI", glXMakeCurrentReadSGI == NULL);
}

#endif /* GLX_SGI_make_current_read */

#ifdef GLX_SGI_swap_control

static void _glewInfo_GLX_SGI_swap_control (void)
{
  GLboolean fi = glewPrintExt("GLX_SGI_swap_control", GLXEW_SGI_swap_control, glxewIsSupported("GLX_SGI_swap_control"), glxewGetExtension("GLX_SGI_swap_control"));

  glewInfoFunc(fi, "glXSwapIntervalSGI", glXSwapIntervalSGI == NULL);
}

#endif /* GLX_SGI_swap_control */

#ifdef GLX_SGI_video_sync

static void _glewInfo_GLX_SGI_video_sync (void)
{
  GLboolean fi = glewPrintExt("GLX_SGI_video_sync", GLXEW_SGI_video_sync, glxewIsSupported("GLX_SGI_video_sync"), glxewGetExtension("GLX_SGI_video_sync"));

  glewInfoFunc(fi, "glXGetVideoSyncSGI", glXGetVideoSyncSGI == NULL);
  glewInfoFunc(fi, "glXWaitVideoSyncSGI", glXWaitVideoSyncSGI == NULL);
}

#endif /* GLX_SGI_video_sync */

#ifdef GLX_SUN_get_transparent_index

static void _glewInfo_GLX_SUN_get_transparent_index (void)
{
  GLboolean fi = glewPrintExt("GLX_SUN_get_transparent_index", GLXEW_SUN_get_transparent_index, glxewIsSupported("GLX_SUN_get_transparent_index"), glxewGetExtension("GLX_SUN_get_transparent_index"));

  glewInfoFunc(fi, "glXGetTransparentIndexSUN", glXGetTransparentIndexSUN == NULL);
}

#endif /* GLX_SUN_get_transparent_index */

#ifdef GLX_SUN_video_resize

static void _glewInfo_GLX_SUN_video_resize (void)
{
  GLboolean fi = glewPrintExt("GLX_SUN_video_resize", GLXEW_SUN_video_resize, glxewIsSupported("GLX_SUN_video_resize"), glxewGetExtension("GLX_SUN_video_resize"));

  glewInfoFunc(fi, "glXGetVideoResizeSUN", glXGetVideoResizeSUN == NULL);
  glewInfoFunc(fi, "glXVideoResizeSUN", glXVideoResizeSUN == NULL);
}

#endif /* GLX_SUN_video_resize */

#endif /* _WIN32 */

/* ------------------------------------------------------------------------ */

static void glewInfo (void)
{
#ifdef GL_VERSION_1_1
  _glewInfo_GL_VERSION_1_1();
#endif /* GL_VERSION_1_1 */
#ifdef GL_VERSION_1_2
  _glewInfo_GL_VERSION_1_2();
#endif /* GL_VERSION_1_2 */
#ifdef GL_VERSION_1_2_1
  _glewInfo_GL_VERSION_1_2_1();
#endif /* GL_VERSION_1_2_1 */
#ifdef GL_VERSION_1_3
  _glewInfo_GL_VERSION_1_3();
#endif /* GL_VERSION_1_3 */
#ifdef GL_VERSION_1_4
  _glewInfo_GL_VERSION_1_4();
#endif /* GL_VERSION_1_4 */
#ifdef GL_VERSION_1_5
  _glewInfo_GL_VERSION_1_5();
#endif /* GL_VERSION_1_5 */
#ifdef GL_VERSION_2_0
  _glewInfo_GL_VERSION_2_0();
#endif /* GL_VERSION_2_0 */
#ifdef GL_VERSION_2_1
  _glewInfo_GL_VERSION_2_1();
#endif /* GL_VERSION_2_1 */
#ifdef GL_VERSION_3_0
  _glewInfo_GL_VERSION_3_0();
#endif /* GL_VERSION_3_0 */
#ifdef GL_VERSION_3_1
  _glewInfo_GL_VERSION_3_1();
#endif /* GL_VERSION_3_1 */
#ifdef GL_VERSION_3_2
  _glewInfo_GL_VERSION_3_2();
#endif /* GL_VERSION_3_2 */
#ifdef GL_VERSION_3_3
  _glewInfo_GL_VERSION_3_3();
#endif /* GL_VERSION_3_3 */
#ifdef GL_VERSION_4_0
  _glewInfo_GL_VERSION_4_0();
#endif /* GL_VERSION_4_0 */
#ifdef GL_VERSION_4_1
  _glewInfo_GL_VERSION_4_1();
#endif /* GL_VERSION_4_1 */
#ifdef GL_VERSION_4_2
  _glewInfo_GL_VERSION_4_2();
#endif /* GL_VERSION_4_2 */
#ifdef GL_VERSION_4_3
  _glewInfo_GL_VERSION_4_3();
#endif /* GL_VERSION_4_3 */
#ifdef GL_VERSION_4_4
  _glewInfo_GL_VERSION_4_4();
#endif /* GL_VERSION_4_4 */
#ifdef GL_VERSION_4_5
  _glewInfo_GL_VERSION_4_5();
#endif /* GL_VERSION_4_5 */
#ifdef GL_VERSION_4_6
  _glewInfo_GL_VERSION_4_6();
#endif /* GL_VERSION_4_6 */
#ifdef GL_3DFX_multisample
  _glewInfo_GL_3DFX_multisample();
#endif /* GL_3DFX_multisample */
#ifdef GL_3DFX_tbuffer
  _glewInfo_GL_3DFX_tbuffer();
#endif /* GL_3DFX_tbuffer */
#ifdef GL_3DFX_texture_compression_FXT1
  _glewInfo_GL_3DFX_texture_compression_FXT1();
#endif /* GL_3DFX_texture_compression_FXT1 */
#ifdef GL_AMD_blend_minmax_factor
  _glewInfo_GL_AMD_blend_minmax_factor();
#endif /* GL_AMD_blend_minmax_factor */
#ifdef GL_AMD_compressed_3DC_texture
  _glewInfo_GL_AMD_compressed_3DC_texture();
#endif /* GL_AMD_compressed_3DC_texture */
#ifdef GL_AMD_compressed_ATC_texture
  _glewInfo_GL_AMD_compressed_ATC_texture();
#endif /* GL_AMD_compressed_ATC_texture */
#ifdef GL_AMD_conservative_depth
  _glewInfo_GL_AMD_conservative_depth();
#endif /* GL_AMD_conservative_depth */
#ifdef GL_AMD_debug_output
  _glewInfo_GL_AMD_debug_output();
#endif /* GL_AMD_debug_output */
#ifdef GL_AMD_depth_clamp_separate
  _glewInfo_GL_AMD_depth_clamp_separate();
#endif /* GL_AMD_depth_clamp_separate */
#ifdef GL_AMD_draw_buffers_blend
  _glewInfo_GL_AMD_draw_buffers_blend();
#endif /* GL_AMD_draw_buffers_blend */
#ifdef GL_AMD_framebuffer_multisample_advanced
  _glewInfo_GL_AMD_framebuffer_multisample_advanced();
#endif /* GL_AMD_framebuffer_multisample_advanced */
#ifdef GL_AMD_framebuffer_sample_positions
  _glewInfo_GL_AMD_framebuffer_sample_positions();
#endif /* GL_AMD_framebuffer_sample_positions */
#ifdef GL_AMD_gcn_shader
  _glewInfo_GL_AMD_gcn_shader();
#endif /* GL_AMD_gcn_shader */
#ifdef GL_AMD_gpu_shader_half_float
  _glewInfo_GL_AMD_gpu_shader_half_float();
#endif /* GL_AMD_gpu_shader_half_float */
#ifdef GL_AMD_gpu_shader_half_float_fetch
  _glewInfo_GL_AMD_gpu_shader_half_float_fetch();
#endif /* GL_AMD_gpu_shader_half_float_fetch */
#ifdef GL_AMD_gpu_shader_int16
  _glewInfo_GL_AMD_gpu_shader_int16();
#endif /* GL_AMD_gpu_shader_int16 */
#ifdef GL_AMD_gpu_shader_int64
  _glewInfo_GL_AMD_gpu_shader_int64();
#endif /* GL_AMD_gpu_shader_int64 */
#ifdef GL_AMD_interleaved_elements
  _glewInfo_GL_AMD_interleaved_elements();
#endif /* GL_AMD_interleaved_elements */
#ifdef GL_AMD_multi_draw_indirect
  _glewInfo_GL_AMD_multi_draw_indirect();
#endif /* GL_AMD_multi_draw_indirect */
#ifdef GL_AMD_name_gen_delete
  _glewInfo_GL_AMD_name_gen_delete();
#endif /* GL_AMD_name_gen_delete */
#ifdef GL_AMD_occlusion_query_event
  _glewInfo_GL_AMD_occlusion_query_event();
#endif /* GL_AMD_occlusion_query_event */
#ifdef GL_AMD_performance_monitor
  _glewInfo_GL_AMD_performance_monitor();
#endif /* GL_AMD_performance_monitor */
#ifdef GL_AMD_pinned_memory
  _glewInfo_GL_AMD_pinned_memory();
#endif /* GL_AMD_pinned_memory */
#ifdef GL_AMD_program_binary_Z400
  _glewInfo_GL_AMD_program_binary_Z400();
#endif /* GL_AMD_program_binary_Z400 */
#ifdef GL_AMD_query_buffer_object
  _glewInfo_GL_AMD_query_buffer_object();
#endif /* GL_AMD_query_buffer_object */
#ifdef GL_AMD_sample_positions
  _glewInfo_GL_AMD_sample_positions();
#endif /* GL_AMD_sample_positions */
#ifdef GL_AMD_seamless_cubemap_per_texture
  _glewInfo_GL_AMD_seamless_cubemap_per_texture();
#endif /* GL_AMD_seamless_cubemap_per_texture */
#ifdef GL_AMD_shader_atomic_counter_ops
  _glewInfo_GL_AMD_shader_atomic_counter_ops();
#endif /* GL_AMD_shader_atomic_counter_ops */
#ifdef GL_AMD_shader_ballot
  _glewInfo_GL_AMD_shader_ballot();
#endif /* GL_AMD_shader_ballot */
#ifdef GL_AMD_shader_explicit_vertex_parameter
  _glewInfo_GL_AMD_shader_explicit_vertex_parameter();
#endif /* GL_AMD_shader_explicit_vertex_parameter */
#ifdef GL_AMD_shader_image_load_store_lod
  _glewInfo_GL_AMD_shader_image_load_store_lod();
#endif /* GL_AMD_shader_image_load_store_lod */
#ifdef GL_AMD_shader_stencil_export
  _glewInfo_GL_AMD_shader_stencil_export();
#endif /* GL_AMD_shader_stencil_export */
#ifdef GL_AMD_shader_stencil_value_export
  _glewInfo_GL_AMD_shader_stencil_value_export();
#endif /* GL_AMD_shader_stencil_value_export */
#ifdef GL_AMD_shader_trinary_minmax
  _glewInfo_GL_AMD_shader_trinary_minmax();
#endif /* GL_AMD_shader_trinary_minmax */
#ifdef GL_AMD_sparse_texture
  _glewInfo_GL_AMD_sparse_texture();
#endif /* GL_AMD_sparse_texture */
#ifdef GL_AMD_stencil_operation_extended
  _glewInfo_GL_AMD_stencil_operation_extended();
#endif /* GL_AMD_stencil_operation_extended */
#ifdef GL_AMD_texture_gather_bias_lod
  _glewInfo_GL_AMD_texture_gather_bias_lod();
#endif /* GL_AMD_texture_gather_bias_lod */
#ifdef GL_AMD_texture_texture4
  _glewInfo_GL_AMD_texture_texture4();
#endif /* GL_AMD_texture_texture4 */
#ifdef GL_AMD_transform_feedback3_lines_triangles
  _glewInfo_GL_AMD_transform_feedback3_lines_triangles();
#endif /* GL_AMD_transform_feedback3_lines_triangles */
#ifdef GL_AMD_transform_feedback4
  _glewInfo_GL_AMD_transform_feedback4();
#endif /* GL_AMD_transform_feedback4 */
#ifdef GL_AMD_vertex_shader_layer
  _glewInfo_GL_AMD_vertex_shader_layer();
#endif /* GL_AMD_vertex_shader_layer */
#ifdef GL_AMD_vertex_shader_tessellator
  _glewInfo_GL_AMD_vertex_shader_tessellator();
#endif /* GL_AMD_vertex_shader_tessellator */
#ifdef GL_AMD_vertex_shader_viewport_index
  _glewInfo_GL_AMD_vertex_shader_viewport_index();
#endif /* GL_AMD_vertex_shader_viewport_index */
#ifdef GL_ANDROID_extension_pack_es31a
  _glewInfo_GL_ANDROID_extension_pack_es31a();
#endif /* GL_ANDROID_extension_pack_es31a */
#ifdef GL_ANGLE_depth_texture
  _glewInfo_GL_ANGLE_depth_texture();
#endif /* GL_ANGLE_depth_texture */
#ifdef GL_ANGLE_framebuffer_blit
  _glewInfo_GL_ANGLE_framebuffer_blit();
#endif /* GL_ANGLE_framebuffer_blit */
#ifdef GL_ANGLE_framebuffer_multisample
  _glewInfo_GL_ANGLE_framebuffer_multisample();
#endif /* GL_ANGLE_framebuffer_multisample */
#ifdef GL_ANGLE_instanced_arrays
  _glewInfo_GL_ANGLE_instanced_arrays();
#endif /* GL_ANGLE_instanced_arrays */
#ifdef GL_ANGLE_pack_reverse_row_order
  _glewInfo_GL_ANGLE_pack_reverse_row_order();
#endif /* GL_ANGLE_pack_reverse_row_order */
#ifdef GL_ANGLE_program_binary
  _glewInfo_GL_ANGLE_program_binary();
#endif /* GL_ANGLE_program_binary */
#ifdef GL_ANGLE_texture_compression_dxt1
  _glewInfo_GL_ANGLE_texture_compression_dxt1();
#endif /* GL_ANGLE_texture_compression_dxt1 */
#ifdef GL_ANGLE_texture_compression_dxt3
  _glewInfo_GL_ANGLE_texture_compression_dxt3();
#endif /* GL_ANGLE_texture_compression_dxt3 */
#ifdef GL_ANGLE_texture_compression_dxt5
  _glewInfo_GL_ANGLE_texture_compression_dxt5();
#endif /* GL_ANGLE_texture_compression_dxt5 */
#ifdef GL_ANGLE_texture_usage
  _glewInfo_GL_ANGLE_texture_usage();
#endif /* GL_ANGLE_texture_usage */
#ifdef GL_ANGLE_timer_query
  _glewInfo_GL_ANGLE_timer_query();
#endif /* GL_ANGLE_timer_query */
#ifdef GL_ANGLE_translated_shader_source
  _glewInfo_GL_ANGLE_translated_shader_source();
#endif /* GL_ANGLE_translated_shader_source */
#ifdef GL_APPLE_aux_depth_stencil
  _glewInfo_GL_APPLE_aux_depth_stencil();
#endif /* GL_APPLE_aux_depth_stencil */
#ifdef GL_APPLE_client_storage
  _glewInfo_GL_APPLE_client_storage();
#endif /* GL_APPLE_client_storage */
#ifdef GL_APPLE_clip_distance
  _glewInfo_GL_APPLE_clip_distance();
#endif /* GL_APPLE_clip_distance */
#ifdef GL_APPLE_color_buffer_packed_float
  _glewInfo_GL_APPLE_color_buffer_packed_float();
#endif /* GL_APPLE_color_buffer_packed_float */
#ifdef GL_APPLE_copy_texture_levels
  _glewInfo_GL_APPLE_copy_texture_levels();
#endif /* GL_APPLE_copy_texture_levels */
#ifdef GL_APPLE_element_array
  _glewInfo_GL_APPLE_element_array();
#endif /* GL_APPLE_element_array */
#ifdef GL_APPLE_fence
  _glewInfo_GL_APPLE_fence();
#endif /* GL_APPLE_fence */
#ifdef GL_APPLE_float_pixels
  _glewInfo_GL_APPLE_float_pixels();
#endif /* GL_APPLE_float_pixels */
#ifdef GL_APPLE_flush_buffer_range
  _glewInfo_GL_APPLE_flush_buffer_range();
#endif /* GL_APPLE_flush_buffer_range */
#ifdef GL_APPLE_framebuffer_multisample
  _glewInfo_GL_APPLE_framebuffer_multisample();
#endif /* GL_APPLE_framebuffer_multisample */
#ifdef GL_APPLE_object_purgeable
  _glewInfo_GL_APPLE_object_purgeable();
#endif /* GL_APPLE_object_purgeable */
#ifdef GL_APPLE_pixel_buffer
  _glewInfo_GL_APPLE_pixel_buffer();
#endif /* GL_APPLE_pixel_buffer */
#ifdef GL_APPLE_rgb_422
  _glewInfo_GL_APPLE_rgb_422();
#endif /* GL_APPLE_rgb_422 */
#ifdef GL_APPLE_row_bytes
  _glewInfo_GL_APPLE_row_bytes();
#endif /* GL_APPLE_row_bytes */
#ifdef GL_APPLE_specular_vector
  _glewInfo_GL_APPLE_specular_vector();
#endif /* GL_APPLE_specular_vector */
#ifdef GL_APPLE_sync
  _glewInfo_GL_APPLE_sync();
#endif /* GL_APPLE_sync */
#ifdef GL_APPLE_texture_2D_limited_npot
  _glewInfo_GL_APPLE_texture_2D_limited_npot();
#endif /* GL_APPLE_texture_2D_limited_npot */
#ifdef GL_APPLE_texture_format_BGRA8888
  _glewInfo_GL_APPLE_texture_format_BGRA8888();
#endif /* GL_APPLE_texture_format_BGRA8888 */
#ifdef GL_APPLE_texture_max_level
  _glewInfo_GL_APPLE_texture_max_level();
#endif /* GL_APPLE_texture_max_level */
#ifdef GL_APPLE_texture_packed_float
  _glewInfo_GL_APPLE_texture_packed_float();
#endif /* GL_APPLE_texture_packed_float */
#ifdef GL_APPLE_texture_range
  _glewInfo_GL_APPLE_texture_range();
#endif /* GL_APPLE_texture_range */
#ifdef GL_APPLE_transform_hint
  _glewInfo_GL_APPLE_transform_hint();
#endif /* GL_APPLE_transform_hint */
#ifdef GL_APPLE_vertex_array_object
  _glewInfo_GL_APPLE_vertex_array_object();
#endif /* GL_APPLE_vertex_array_object */
#ifdef GL_APPLE_vertex_array_range
  _glewInfo_GL_APPLE_vertex_array_range();
#endif /* GL_APPLE_vertex_array_range */
#ifdef GL_APPLE_vertex_program_evaluators
  _glewInfo_GL_APPLE_vertex_program_evaluators();
#endif /* GL_APPLE_vertex_program_evaluators */
#ifdef GL_APPLE_ycbcr_422
  _glewInfo_GL_APPLE_ycbcr_422();
#endif /* GL_APPLE_ycbcr_422 */
#ifdef GL_ARB_ES2_compatibility
  _glewInfo_GL_ARB_ES2_compatibility();
#endif /* GL_ARB_ES2_compatibility */
#ifdef GL_ARB_ES3_1_compatibility
  _glewInfo_GL_ARB_ES3_1_compatibility();
#endif /* GL_ARB_ES3_1_compatibility */
#ifdef GL_ARB_ES3_2_compatibility
  _glewInfo_GL_ARB_ES3_2_compatibility();
#endif /* GL_ARB_ES3_2_compatibility */
#ifdef GL_ARB_ES3_compatibility
  _glewInfo_GL_ARB_ES3_compatibility();
#endif /* GL_ARB_ES3_compatibility */
#ifdef GL_ARB_arrays_of_arrays
  _glewInfo_GL_ARB_arrays_of_arrays();
#endif /* GL_ARB_arrays_of_arrays */
#ifdef GL_ARB_base_instance
  _glewInfo_GL_ARB_base_instance();
#endif /* GL_ARB_base_instance */
#ifdef GL_ARB_bindless_texture
  _glewInfo_GL_ARB_bindless_texture();
#endif /* GL_ARB_bindless_texture */
#ifdef GL_ARB_blend_func_extended
  _glewInfo_GL_ARB_blend_func_extended();
#endif /* GL_ARB_blend_func_extended */
#ifdef GL_ARB_buffer_storage
  _glewInfo_GL_ARB_buffer_storage();
#endif /* GL_ARB_buffer_storage */
#ifdef GL_ARB_cl_event
  _glewInfo_GL_ARB_cl_event();
#endif /* GL_ARB_cl_event */
#ifdef GL_ARB_clear_buffer_object
  _glewInfo_GL_ARB_clear_buffer_object();
#endif /* GL_ARB_clear_buffer_object */
#ifdef GL_ARB_clear_texture
  _glewInfo_GL_ARB_clear_texture();
#endif /* GL_ARB_clear_texture */
#ifdef GL_ARB_clip_control
  _glewInfo_GL_ARB_clip_control();
#endif /* GL_ARB_clip_control */
#ifdef GL_ARB_color_buffer_float
  _glewInfo_GL_ARB_color_buffer_float();
#endif /* GL_ARB_color_buffer_float */
#ifdef GL_ARB_compatibility
  _glewInfo_GL_ARB_compatibility();
#endif /* GL_ARB_compatibility */
#ifdef GL_ARB_compressed_texture_pixel_storage
  _glewInfo_GL_ARB_compressed_texture_pixel_storage();
#endif /* GL_ARB_compressed_texture_pixel_storage */
#ifdef GL_ARB_compute_shader
  _glewInfo_GL_ARB_compute_shader();
#endif /* GL_ARB_compute_shader */
#ifdef GL_ARB_compute_variable_group_size
  _glewInfo_GL_ARB_compute_variable_group_size();
#endif /* GL_ARB_compute_variable_group_size */
#ifdef GL_ARB_conditional_render_inverted
  _glewInfo_GL_ARB_conditional_render_inverted();
#endif /* GL_ARB_conditional_render_inverted */
#ifdef GL_ARB_conservative_depth
  _glewInfo_GL_ARB_conservative_depth();
#endif /* GL_ARB_conservative_depth */
#ifdef GL_ARB_copy_buffer
  _glewInfo_GL_ARB_copy_buffer();
#endif /* GL_ARB_copy_buffer */
#ifdef GL_ARB_copy_image
  _glewInfo_GL_ARB_copy_image();
#endif /* GL_ARB_copy_image */
#ifdef GL_ARB_cull_distance
  _glewInfo_GL_ARB_cull_distance();
#endif /* GL_ARB_cull_distance */
#ifdef GL_ARB_debug_output
  _glewInfo_GL_ARB_debug_output();
#endif /* GL_ARB_debug_output */
#ifdef GL_ARB_depth_buffer_float
  _glewInfo_GL_ARB_depth_buffer_float();
#endif /* GL_ARB_depth_buffer_float */
#ifdef GL_ARB_depth_clamp
  _glewInfo_GL_ARB_depth_clamp();
#endif /* GL_ARB_depth_clamp */
#ifdef GL_ARB_depth_texture
  _glewInfo_GL_ARB_depth_texture();
#endif /* GL_ARB_depth_texture */
#ifdef GL_ARB_derivative_control
  _glewInfo_GL_ARB_derivative_control();
#endif /* GL_ARB_derivative_control */
#ifdef GL_ARB_direct_state_access
  _glewInfo_GL_ARB_direct_state_access();
#endif /* GL_ARB_direct_state_access */
#ifdef GL_ARB_draw_buffers
  _glewInfo_GL_ARB_draw_buffers();
#endif /* GL_ARB_draw_buffers */
#ifdef GL_ARB_draw_buffers_blend
  _glewInfo_GL_ARB_draw_buffers_blend();
#endif /* GL_ARB_draw_buffers_blend */
#ifdef GL_ARB_draw_elements_base_vertex
  _glewInfo_GL_ARB_draw_elements_base_vertex();
#endif /* GL_ARB_draw_elements_base_vertex */
#ifdef GL_ARB_draw_indirect
  _glewInfo_GL_ARB_draw_indirect();
#endif /* GL_ARB_draw_indirect */
#ifdef GL_ARB_draw_instanced
  _glewInfo_GL_ARB_draw_instanced();
#endif /* GL_ARB_draw_instanced */
#ifdef GL_ARB_enhanced_layouts
  _glewInfo_GL_ARB_enhanced_layouts();
#endif /* GL_ARB_enhanced_layouts */
#ifdef GL_ARB_explicit_attrib_location
  _glewInfo_GL_ARB_explicit_attrib_location();
#endif /* GL_ARB_explicit_attrib_location */
#ifdef GL_ARB_explicit_uniform_location
  _glewInfo_GL_ARB_explicit_uniform_location();
#endif /* GL_ARB_explicit_uniform_location */
#ifdef GL_ARB_fragment_coord_conventions
  _glewInfo_GL_ARB_fragment_coord_conventions();
#endif /* GL_ARB_fragment_coord_conventions */
#ifdef GL_ARB_fragment_layer_viewport
  _glewInfo_GL_ARB_fragment_layer_viewport();
#endif /* GL_ARB_fragment_layer_viewport */
#ifdef GL_ARB_fragment_program
  _glewInfo_GL_ARB_fragment_program();
#endif /* GL_ARB_fragment_program */
#ifdef GL_ARB_fragment_program_shadow
  _glewInfo_GL_ARB_fragment_program_shadow();
#endif /* GL_ARB_fragment_program_shadow */
#ifdef GL_ARB_fragment_shader
  _glewInfo_GL_ARB_fragment_shader();
#endif /* GL_ARB_fragment_shader */
#ifdef GL_ARB_fragment_shader_interlock
  _glewInfo_GL_ARB_fragment_shader_interlock();
#endif /* GL_ARB_fragment_shader_interlock */
#ifdef GL_ARB_framebuffer_no_attachments
  _glewInfo_GL_ARB_framebuffer_no_attachments();
#endif /* GL_ARB_framebuffer_no_attachments */
#ifdef GL_ARB_framebuffer_object
  _glewInfo_GL_ARB_framebuffer_object();
#endif /* GL_ARB_framebuffer_object */
#ifdef GL_ARB_framebuffer_sRGB
  _glewInfo_GL_ARB_framebuffer_sRGB();
#endif /* GL_ARB_framebuffer_sRGB */
#ifdef GL_ARB_geometry_shader4
  _glewInfo_GL_ARB_geometry_shader4();
#endif /* GL_ARB_geometry_shader4 */
#ifdef GL_ARB_get_program_binary
  _glewInfo_GL_ARB_get_program_binary();
#endif /* GL_ARB_get_program_binary */
#ifdef GL_ARB_get_texture_sub_image
  _glewInfo_GL_ARB_get_texture_sub_image();
#endif /* GL_ARB_get_texture_sub_image */
#ifdef GL_ARB_gl_spirv
  _glewInfo_GL_ARB_gl_spirv();
#endif /* GL_ARB_gl_spirv */
#ifdef GL_ARB_gpu_shader5
  _glewInfo_GL_ARB_gpu_shader5();
#endif /* GL_ARB_gpu_shader5 */
#ifdef GL_ARB_gpu_shader_fp64
  _glewInfo_GL_ARB_gpu_shader_fp64();
#endif /* GL_ARB_gpu_shader_fp64 */
#ifdef GL_ARB_gpu_shader_int64
  _glewInfo_GL_ARB_gpu_shader_int64();
#endif /* GL_ARB_gpu_shader_int64 */
#ifdef GL_ARB_half_float_pixel
  _glewInfo_GL_ARB_half_float_pixel();
#endif /* GL_ARB_half_float_pixel */
#ifdef GL_ARB_half_float_vertex
  _glewInfo_GL_ARB_half_float_vertex();
#endif /* GL_ARB_half_float_vertex */
#ifdef GL_ARB_imaging
  _glewInfo_GL_ARB_imaging();
#endif /* GL_ARB_imaging */
#ifdef GL_ARB_indirect_parameters
  _glewInfo_GL_ARB_indirect_parameters();
#endif /* GL_ARB_indirect_parameters */
#ifdef GL_ARB_instanced_arrays
  _glewInfo_GL_ARB_instanced_arrays();
#endif /* GL_ARB_instanced_arrays */
#ifdef GL_ARB_internalformat_query
  _glewInfo_GL_ARB_internalformat_query();
#endif /* GL_ARB_internalformat_query */
#ifdef GL_ARB_internalformat_query2
  _glewInfo_GL_ARB_internalformat_query2();
#endif /* GL_ARB_internalformat_query2 */
#ifdef GL_ARB_invalidate_subdata
  _glewInfo_GL_ARB_invalidate_subdata();
#endif /* GL_ARB_invalidate_subdata */
#ifdef GL_ARB_map_buffer_alignment
  _glewInfo_GL_ARB_map_buffer_alignment();
#endif /* GL_ARB_map_buffer_alignment */
#ifdef GL_ARB_map_buffer_range
  _glewInfo_GL_ARB_map_buffer_range();
#endif /* GL_ARB_map_buffer_range */
#ifdef GL_ARB_matrix_palette
  _glewInfo_GL_ARB_matrix_palette();
#endif /* GL_ARB_matrix_palette */
#ifdef GL_ARB_multi_bind
  _glewInfo_GL_ARB_multi_bind();
#endif /* GL_ARB_multi_bind */
#ifdef GL_ARB_multi_draw_indirect
  _glewInfo_GL_ARB_multi_draw_indirect();
#endif /* GL_ARB_multi_draw_indirect */
#ifdef GL_ARB_multisample
  _glewInfo_GL_ARB_multisample();
#endif /* GL_ARB_multisample */
#ifdef GL_ARB_multitexture
  _glewInfo_GL_ARB_multitexture();
#endif /* GL_ARB_multitexture */
#ifdef GL_ARB_occlusion_query
  _glewInfo_GL_ARB_occlusion_query();
#endif /* GL_ARB_occlusion_query */
#ifdef GL_ARB_occlusion_query2
  _glewInfo_GL_ARB_occlusion_query2();
#endif /* GL_ARB_occlusion_query2 */
#ifdef GL_ARB_parallel_shader_compile
  _glewInfo_GL_ARB_parallel_shader_compile();
#endif /* GL_ARB_parallel_shader_compile */
#ifdef GL_ARB_pipeline_statistics_query
  _glewInfo_GL_ARB_pipeline_statistics_query();
#endif /* GL_ARB_pipeline_statistics_query */
#ifdef GL_ARB_pixel_buffer_object
  _glewInfo_GL_ARB_pixel_buffer_object();
#endif /* GL_ARB_pixel_buffer_object */
#ifdef GL_ARB_point_parameters
  _glewInfo_GL_ARB_point_parameters();
#endif /* GL_ARB_point_parameters */
#ifdef GL_ARB_point_sprite
  _glewInfo_GL_ARB_point_sprite();
#endif /* GL_ARB_point_sprite */
#ifdef GL_ARB_polygon_offset_clamp
  _glewInfo_GL_ARB_polygon_offset_clamp();
#endif /* GL_ARB_polygon_offset_clamp */
#ifdef GL_ARB_post_depth_coverage
  _glewInfo_GL_ARB_post_depth_coverage();
#endif /* GL_ARB_post_depth_coverage */
#ifdef GL_ARB_program_interface_query
  _glewInfo_GL_ARB_program_interface_query();
#endif /* GL_ARB_program_interface_query */
#ifdef GL_ARB_provoking_vertex
  _glewInfo_GL_ARB_provoking_vertex();
#endif /* GL_ARB_provoking_vertex */
#ifdef GL_ARB_query_buffer_object
  _glewInfo_GL_ARB_query_buffer_object();
#endif /* GL_ARB_query_buffer_object */
#ifdef GL_ARB_robust_buffer_access_behavior
  _glewInfo_GL_ARB_robust_buffer_access_behavior();
#endif /* GL_ARB_robust_buffer_access_behavior */
#ifdef GL_ARB_robustness
  _glewInfo_GL_ARB_robustness();
#endif /* GL_ARB_robustness */
#ifdef GL_ARB_robustness_application_isolation
  _glewInfo_GL_ARB_robustness_application_isolation();
#endif /* GL_ARB_robustness_application_isolation */
#ifdef GL_ARB_robustness_share_group_isolation
  _glewInfo_GL_ARB_robustness_share_group_isolation();
#endif /* GL_ARB_robustness_share_group_isolation */
#ifdef GL_ARB_sample_locations
  _glewInfo_GL_ARB_sample_locations();
#endif /* GL_ARB_sample_locations */
#ifdef GL_ARB_sample_shading
  _glewInfo_GL_ARB_sample_shading();
#endif /* GL_ARB_sample_shading */
#ifdef GL_ARB_sampler_objects
  _glewInfo_GL_ARB_sampler_objects();
#endif /* GL_ARB_sampler_objects */
#ifdef GL_ARB_seamless_cube_map
  _glewInfo_GL_ARB_seamless_cube_map();
#endif /* GL_ARB_seamless_cube_map */
#ifdef GL_ARB_seamless_cubemap_per_texture
  _glewInfo_GL_ARB_seamless_cubemap_per_texture();
#endif /* GL_ARB_seamless_cubemap_per_texture */
#ifdef GL_ARB_separate_shader_objects
  _glewInfo_GL_ARB_separate_shader_objects();
#endif /* GL_ARB_separate_shader_objects */
#ifdef GL_ARB_shader_atomic_counter_ops
  _glewInfo_GL_ARB_shader_atomic_counter_ops();
#endif /* GL_ARB_shader_atomic_counter_ops */
#ifdef GL_ARB_shader_atomic_counters
  _glewInfo_GL_ARB_shader_atomic_counters();
#endif /* GL_ARB_shader_atomic_counters */
#ifdef GL_ARB_shader_ballot
  _glewInfo_GL_ARB_shader_ballot();
#endif /* GL_ARB_shader_ballot */
#ifdef GL_ARB_shader_bit_encoding
  _glewInfo_GL_ARB_shader_bit_encoding();
#endif /* GL_ARB_shader_bit_encoding */
#ifdef GL_ARB_shader_clock
  _glewInfo_GL_ARB_shader_clock();
#endif /* GL_ARB_shader_clock */
#ifdef GL_ARB_shader_draw_parameters
  _glewInfo_GL_ARB_shader_draw_parameters();
#endif /* GL_ARB_shader_draw_parameters */
#ifdef GL_ARB_shader_group_vote
  _glewInfo_GL_ARB_shader_group_vote();
#endif /* GL_ARB_shader_group_vote */
#ifdef GL_ARB_shader_image_load_store
  _glewInfo_GL_ARB_shader_image_load_store();
#endif /* GL_ARB_shader_image_load_store */
#ifdef GL_ARB_shader_image_size
  _glewInfo_GL_ARB_shader_image_size();
#endif /* GL_ARB_shader_image_size */
#ifdef GL_ARB_shader_objects
  _glewInfo_GL_ARB_shader_objects();
#endif /* GL_ARB_shader_objects */
#ifdef GL_ARB_shader_precision
  _glewInfo_GL_ARB_shader_precision();
#endif /* GL_ARB_shader_precision */
#ifdef GL_ARB_shader_stencil_export
  _glewInfo_GL_ARB_shader_stencil_export();
#endif /* GL_ARB_shader_stencil_export */
#ifdef GL_ARB_shader_storage_buffer_object
  _glewInfo_GL_ARB_shader_storage_buffer_object();
#endif /* GL_ARB_shader_storage_buffer_object */
#ifdef GL_ARB_shader_subroutine
  _glewInfo_GL_ARB_shader_subroutine();
#endif /* GL_ARB_shader_subroutine */
#ifdef GL_ARB_shader_texture_image_samples
  _glewInfo_GL_ARB_shader_texture_image_samples();
#endif /* GL_ARB_shader_texture_image_samples */
#ifdef GL_ARB_shader_texture_lod
  _glewInfo_GL_ARB_shader_texture_lod();
#endif /* GL_ARB_shader_texture_lod */
#ifdef GL_ARB_shader_viewport_layer_array
  _glewInfo_GL_ARB_shader_viewport_layer_array();
#endif /* GL_ARB_shader_viewport_layer_array */
#ifdef GL_ARB_shading_language_100
  _glewInfo_GL_ARB_shading_language_100();
#endif /* GL_ARB_shading_language_100 */
#ifdef GL_ARB_shading_language_420pack
  _glewInfo_GL_ARB_shading_language_420pack();
#endif /* GL_ARB_shading_language_420pack */
#ifdef GL_ARB_shading_language_include
  _glewInfo_GL_ARB_shading_language_include();
#endif /* GL_ARB_shading_language_include */
#ifdef GL_ARB_shading_language_packing
  _glewInfo_GL_ARB_shading_language_packing();
#endif /* GL_ARB_shading_language_packing */
#ifdef GL_ARB_shadow
  _glewInfo_GL_ARB_shadow();
#endif /* GL_ARB_shadow */
#ifdef GL_ARB_shadow_ambient
  _glewInfo_GL_ARB_shadow_ambient();
#endif /* GL_ARB_shadow_ambient */
#ifdef GL_ARB_sparse_buffer
  _glewInfo_GL_ARB_sparse_buffer();
#endif /* GL_ARB_sparse_buffer */
#ifdef GL_ARB_sparse_texture
  _glewInfo_GL_ARB_sparse_texture();
#endif /* GL_ARB_sparse_texture */
#ifdef GL_ARB_sparse_texture2
  _glewInfo_GL_ARB_sparse_texture2();
#endif /* GL_ARB_sparse_texture2 */
#ifdef GL_ARB_sparse_texture_clamp
  _glewInfo_GL_ARB_sparse_texture_clamp();
#endif /* GL_ARB_sparse_texture_clamp */
#ifdef GL_ARB_spirv_extensions
  _glewInfo_GL_ARB_spirv_extensions();
#endif /* GL_ARB_spirv_extensions */
#ifdef GL_ARB_stencil_texturing
  _glewInfo_GL_ARB_stencil_texturing();
#endif /* GL_ARB_stencil_texturing */
#ifdef GL_ARB_sync
  _glewInfo_GL_ARB_sync();
#endif /* GL_ARB_sync */
#ifdef GL_ARB_tessellation_shader
  _glewInfo_GL_ARB_tessellation_shader();
#endif /* GL_ARB_tessellation_shader */
#ifdef GL_ARB_texture_barrier
  _glewInfo_GL_ARB_texture_barrier();
#endif /* GL_ARB_texture_barrier */
#ifdef GL_ARB_texture_border_clamp
  _glewInfo_GL_ARB_texture_border_clamp();
#endif /* GL_ARB_texture_border_clamp */
#ifdef GL_ARB_texture_buffer_object
  _glewInfo_GL_ARB_texture_buffer_object();
#endif /* GL_ARB_texture_buffer_object */
#ifdef GL_ARB_texture_buffer_object_rgb32
  _glewInfo_GL_ARB_texture_buffer_object_rgb32();
#endif /* GL_ARB_texture_buffer_object_rgb32 */
#ifdef GL_ARB_texture_buffer_range
  _glewInfo_GL_ARB_texture_buffer_range();
#endif /* GL_ARB_texture_buffer_range */
#ifdef GL_ARB_texture_compression
  _glewInfo_GL_ARB_texture_compression();
#endif /* GL_ARB_texture_compression */
#ifdef GL_ARB_texture_compression_bptc
  _glewInfo_GL_ARB_texture_compression_bptc();
#endif /* GL_ARB_texture_compression_bptc */
#ifdef GL_ARB_texture_compression_rgtc
  _glewInfo_GL_ARB_texture_compression_rgtc();
#endif /* GL_ARB_texture_compression_rgtc */
#ifdef GL_ARB_texture_cube_map
  _glewInfo_GL_ARB_texture_cube_map();
#endif /* GL_ARB_texture_cube_map */
#ifdef GL_ARB_texture_cube_map_array
  _glewInfo_GL_ARB_texture_cube_map_array();
#endif /* GL_ARB_texture_cube_map_array */
#ifdef GL_ARB_texture_env_add
  _glewInfo_GL_ARB_texture_env_add();
#endif /* GL_ARB_texture_env_add */
#ifdef GL_ARB_texture_env_combine
  _glewInfo_GL_ARB_texture_env_combine();
#endif /* GL_ARB_texture_env_combine */
#ifdef GL_ARB_texture_env_crossbar
  _glewInfo_GL_ARB_texture_env_crossbar();
#endif /* GL_ARB_texture_env_crossbar */
#ifdef GL_ARB_texture_env_dot3
  _glewInfo_GL_ARB_texture_env_dot3();
#endif /* GL_ARB_texture_env_dot3 */
#ifdef GL_ARB_texture_filter_anisotropic
  _glewInfo_GL_ARB_texture_filter_anisotropic();
#endif /* GL_ARB_texture_filter_anisotropic */
#ifdef GL_ARB_texture_filter_minmax
  _glewInfo_GL_ARB_texture_filter_minmax();
#endif /* GL_ARB_texture_filter_minmax */
#ifdef GL_ARB_texture_float
  _glewInfo_GL_ARB_texture_float();
#endif /* GL_ARB_texture_float */
#ifdef GL_ARB_texture_gather
  _glewInfo_GL_ARB_texture_gather();
#endif /* GL_ARB_texture_gather */
#ifdef GL_ARB_texture_mirror_clamp_to_edge
  _glewInfo_GL_ARB_texture_mirror_clamp_to_edge();
#endif /* GL_ARB_texture_mirror_clamp_to_edge */
#ifdef GL_ARB_texture_mirrored_repeat
  _glewInfo_GL_ARB_texture_mirrored_repeat();
#endif /* GL_ARB_texture_mirrored_repeat */
#ifdef GL_ARB_texture_multisample
  _glewInfo_GL_ARB_texture_multisample();
#endif /* GL_ARB_texture_multisample */
#ifdef GL_ARB_texture_non_power_of_two
  _glewInfo_GL_ARB_texture_non_power_of_two();
#endif /* GL_ARB_texture_non_power_of_two */
#ifdef GL_ARB_texture_query_levels
  _glewInfo_GL_ARB_texture_query_levels();
#endif /* GL_ARB_texture_query_levels */
#ifdef GL_ARB_texture_query_lod
  _glewInfo_GL_ARB_texture_query_lod();
#endif /* GL_ARB_texture_query_lod */
#ifdef GL_ARB_texture_rectangle
  _glewInfo_GL_ARB_texture_rectangle();
#endif /* GL_ARB_texture_rectangle */
#ifdef GL_ARB_texture_rg
  _glewInfo_GL_ARB_texture_rg();
#endif /* GL_ARB_texture_rg */
#ifdef GL_ARB_texture_rgb10_a2ui
  _glewInfo_GL_ARB_texture_rgb10_a2ui();
#endif /* GL_ARB_texture_rgb10_a2ui */
#ifdef GL_ARB_texture_stencil8
  _glewInfo_GL_ARB_texture_stencil8();
#endif /* GL_ARB_texture_stencil8 */
#ifdef GL_ARB_texture_storage
  _glewInfo_GL_ARB_texture_storage();
#endif /* GL_ARB_texture_storage */
#ifdef GL_ARB_texture_storage_multisample
  _glewInfo_GL_ARB_texture_storage_multisample();
#endif /* GL_ARB_texture_storage_multisample */
#ifdef GL_ARB_texture_swizzle
  _glewInfo_GL_ARB_texture_swizzle();
#endif /* GL_ARB_texture_swizzle */
#ifdef GL_ARB_texture_view
  _glewInfo_GL_ARB_texture_view();
#endif /* GL_ARB_texture_view */
#ifdef GL_ARB_timer_query
  _glewInfo_GL_ARB_timer_query();
#endif /* GL_ARB_timer_query */
#ifdef GL_ARB_transform_feedback2
  _glewInfo_GL_ARB_transform_feedback2();
#endif /* GL_ARB_transform_feedback2 */
#ifdef GL_ARB_transform_feedback3
  _glewInfo_GL_ARB_transform_feedback3();
#endif /* GL_ARB_transform_feedback3 */
#ifdef GL_ARB_transform_feedback_instanced
  _glewInfo_GL_ARB_transform_feedback_instanced();
#endif /* GL_ARB_transform_feedback_instanced */
#ifdef GL_ARB_transform_feedback_overflow_query
  _glewInfo_GL_ARB_transform_feedback_overflow_query();
#endif /* GL_ARB_transform_feedback_overflow_query */
#ifdef GL_ARB_transpose_matrix
  _glewInfo_GL_ARB_transpose_matrix();
#endif /* GL_ARB_transpose_matrix */
#ifdef GL_ARB_uniform_buffer_object
  _glewInfo_GL_ARB_uniform_buffer_object();
#endif /* GL_ARB_uniform_buffer_object */
#ifdef GL_ARB_vertex_array_bgra
  _glewInfo_GL_ARB_vertex_array_bgra();
#endif /* GL_ARB_vertex_array_bgra */
#ifdef GL_ARB_vertex_array_object
  _glewInfo_GL_ARB_vertex_array_object();
#endif /* GL_ARB_vertex_array_object */
#ifdef GL_ARB_vertex_attrib_64bit
  _glewInfo_GL_ARB_vertex_attrib_64bit();
#endif /* GL_ARB_vertex_attrib_64bit */
#ifdef GL_ARB_vertex_attrib_binding
  _glewInfo_GL_ARB_vertex_attrib_binding();
#endif /* GL_ARB_vertex_attrib_binding */
#ifdef GL_ARB_vertex_blend
  _glewInfo_GL_ARB_vertex_blend();
#endif /* GL_ARB_vertex_blend */
#ifdef GL_ARB_vertex_buffer_object
  _glewInfo_GL_ARB_vertex_buffer_object();
#endif /* GL_ARB_vertex_buffer_object */
#ifdef GL_ARB_vertex_program
  _glewInfo_GL_ARB_vertex_program();
#endif /* GL_ARB_vertex_program */
#ifdef GL_ARB_vertex_shader
  _glewInfo_GL_ARB_vertex_shader();
#endif /* GL_ARB_vertex_shader */
#ifdef GL_ARB_vertex_type_10f_11f_11f_rev
  _glewInfo_GL_ARB_vertex_type_10f_11f_11f_rev();
#endif /* GL_ARB_vertex_type_10f_11f_11f_rev */
#ifdef GL_ARB_vertex_type_2_10_10_10_rev
  _glewInfo_GL_ARB_vertex_type_2_10_10_10_rev();
#endif /* GL_ARB_vertex_type_2_10_10_10_rev */
#ifdef GL_ARB_viewport_array
  _glewInfo_GL_ARB_viewport_array();
#endif /* GL_ARB_viewport_array */
#ifdef GL_ARB_window_pos
  _glewInfo_GL_ARB_window_pos();
#endif /* GL_ARB_window_pos */
#ifdef GL_ARM_mali_program_binary
  _glewInfo_GL_ARM_mali_program_binary();
#endif /* GL_ARM_mali_program_binary */
#ifdef GL_ARM_mali_shader_binary
  _glewInfo_GL_ARM_mali_shader_binary();
#endif /* GL_ARM_mali_shader_binary */
#ifdef GL_ARM_rgba8
  _glewInfo_GL_ARM_rgba8();
#endif /* GL_ARM_rgba8 */
#ifdef GL_ARM_shader_framebuffer_fetch
  _glewInfo_GL_ARM_shader_framebuffer_fetch();
#endif /* GL_ARM_shader_framebuffer_fetch */
#ifdef GL_ARM_shader_framebuffer_fetch_depth_stencil
  _glewInfo_GL_ARM_shader_framebuffer_fetch_depth_stencil();
#endif /* GL_ARM_shader_framebuffer_fetch_depth_stencil */
#ifdef GL_ARM_texture_unnormalized_coordinates
  _glewInfo_GL_ARM_texture_unnormalized_coordinates();
#endif /* GL_ARM_texture_unnormalized_coordinates */
#ifdef GL_ATIX_point_sprites
  _glewInfo_GL_ATIX_point_sprites();
#endif /* GL_ATIX_point_sprites */
#ifdef GL_ATIX_texture_env_combine3
  _glewInfo_GL_ATIX_texture_env_combine3();
#endif /* GL_ATIX_texture_env_combine3 */
#ifdef GL_ATIX_texture_env_route
  _glewInfo_GL_ATIX_texture_env_route();
#endif /* GL_ATIX_texture_env_route */
#ifdef GL_ATIX_vertex_shader_output_point_size
  _glewInfo_GL_ATIX_vertex_shader_output_point_size();
#endif /* GL_ATIX_vertex_shader_output_point_size */
#ifdef GL_ATI_draw_buffers
  _glewInfo_GL_ATI_draw_buffers();
#endif /* GL_ATI_draw_buffers */
#ifdef GL_ATI_element_array
  _glewInfo_GL_ATI_element_array();
#endif /* GL_ATI_element_array */
#ifdef GL_ATI_envmap_bumpmap
  _glewInfo_GL_ATI_envmap_bumpmap();
#endif /* GL_ATI_envmap_bumpmap */
#ifdef GL_ATI_fragment_shader
  _glewInfo_GL_ATI_fragment_shader();
#endif /* GL_ATI_fragment_shader */
#ifdef GL_ATI_map_object_buffer
  _glewInfo_GL_ATI_map_object_buffer();
#endif /* GL_ATI_map_object_buffer */
#ifdef GL_ATI_meminfo
  _glewInfo_GL_ATI_meminfo();
#endif /* GL_ATI_meminfo */
#ifdef GL_ATI_pn_triangles
  _glewInfo_GL_ATI_pn_triangles();
#endif /* GL_ATI_pn_triangles */
#ifdef GL_ATI_separate_stencil
  _glewInfo_GL_ATI_separate_stencil();
#endif /* GL_ATI_separate_stencil */
#ifdef GL_ATI_shader_texture_lod
  _glewInfo_GL_ATI_shader_texture_lod();
#endif /* GL_ATI_shader_texture_lod */
#ifdef GL_ATI_text_fragment_shader
  _glewInfo_GL_ATI_text_fragment_shader();
#endif /* GL_ATI_text_fragment_shader */
#ifdef GL_ATI_texture_compression_3dc
  _glewInfo_GL_ATI_texture_compression_3dc();
#endif /* GL_ATI_texture_compression_3dc */
#ifdef GL_ATI_texture_env_combine3
  _glewInfo_GL_ATI_texture_env_combine3();
#endif /* GL_ATI_texture_env_combine3 */
#ifdef GL_ATI_texture_float
  _glewInfo_GL_ATI_texture_float();
#endif /* GL_ATI_texture_float */
#ifdef GL_ATI_texture_mirror_once
  _glewInfo_GL_ATI_texture_mirror_once();
#endif /* GL_ATI_texture_mirror_once */
#ifdef GL_ATI_vertex_array_object
  _glewInfo_GL_ATI_vertex_array_object();
#endif /* GL_ATI_vertex_array_object */
#ifdef GL_ATI_vertex_attrib_array_object
  _glewInfo_GL_ATI_vertex_attrib_array_object();
#endif /* GL_ATI_vertex_attrib_array_object */
#ifdef GL_ATI_vertex_streams
  _glewInfo_GL_ATI_vertex_streams();
#endif /* GL_ATI_vertex_streams */
#ifdef GL_DMP_program_binary
  _glewInfo_GL_DMP_program_binary();
#endif /* GL_DMP_program_binary */
#ifdef GL_DMP_shader_binary
  _glewInfo_GL_DMP_shader_binary();
#endif /* GL_DMP_shader_binary */
#ifdef GL_EXT_422_pixels
  _glewInfo_GL_EXT_422_pixels();
#endif /* GL_EXT_422_pixels */
#ifdef GL_EXT_Cg_shader
  _glewInfo_GL_EXT_Cg_shader();
#endif /* GL_EXT_Cg_shader */
#ifdef GL_EXT_EGL_image_array
  _glewInfo_GL_EXT_EGL_image_array();
#endif /* GL_EXT_EGL_image_array */
#ifdef GL_EXT_EGL_image_external_wrap_modes
  _glewInfo_GL_EXT_EGL_image_external_wrap_modes();
#endif /* GL_EXT_EGL_image_external_wrap_modes */
#ifdef GL_EXT_EGL_image_storage
  _glewInfo_GL_EXT_EGL_image_storage();
#endif /* GL_EXT_EGL_image_storage */
#ifdef GL_EXT_EGL_sync
  _glewInfo_GL_EXT_EGL_sync();
#endif /* GL_EXT_EGL_sync */
#ifdef GL_EXT_YUV_target
  _glewInfo_GL_EXT_YUV_target();
#endif /* GL_EXT_YUV_target */
#ifdef GL_EXT_abgr
  _glewInfo_GL_EXT_abgr();
#endif /* GL_EXT_abgr */
#ifdef GL_EXT_base_instance
  _glewInfo_GL_EXT_base_instance();
#endif /* GL_EXT_base_instance */
#ifdef GL_EXT_bgra
  _glewInfo_GL_EXT_bgra();
#endif /* GL_EXT_bgra */
#ifdef GL_EXT_bindable_uniform
  _glewInfo_GL_EXT_bindable_uniform();
#endif /* GL_EXT_bindable_uniform */
#ifdef GL_EXT_blend_color
  _glewInfo_GL_EXT_blend_color();
#endif /* GL_EXT_blend_color */
#ifdef GL_EXT_blend_equation_separate
  _glewInfo_GL_EXT_blend_equation_separate();
#endif /* GL_EXT_blend_equation_separate */
#ifdef GL_EXT_blend_func_extended
  _glewInfo_GL_EXT_blend_func_extended();
#endif /* GL_EXT_blend_func_extended */
#ifdef GL_EXT_blend_func_separate
  _glewInfo_GL_EXT_blend_func_separate();
#endif /* GL_EXT_blend_func_separate */
#ifdef GL_EXT_blend_logic_op
  _glewInfo_GL_EXT_blend_logic_op();
#endif /* GL_EXT_blend_logic_op */
#ifdef GL_EXT_blend_minmax
  _glewInfo_GL_EXT_blend_minmax();
#endif /* GL_EXT_blend_minmax */
#ifdef GL_EXT_blend_subtract
  _glewInfo_GL_EXT_blend_subtract();
#endif /* GL_EXT_blend_subtract */
#ifdef GL_EXT_buffer_storage
  _glewInfo_GL_EXT_buffer_storage();
#endif /* GL_EXT_buffer_storage */
#ifdef GL_EXT_clear_texture
  _glewInfo_GL_EXT_clear_texture();
#endif /* GL_EXT_clear_texture */
#ifdef GL_EXT_clip_control
  _glewInfo_GL_EXT_clip_control();
#endif /* GL_EXT_clip_control */
#ifdef GL_EXT_clip_cull_distance
  _glewInfo_GL_EXT_clip_cull_distance();
#endif /* GL_EXT_clip_cull_distance */
#ifdef GL_EXT_clip_volume_hint
  _glewInfo_GL_EXT_clip_volume_hint();
#endif /* GL_EXT_clip_volume_hint */
#ifdef GL_EXT_cmyka
  _glewInfo_GL_EXT_cmyka();
#endif /* GL_EXT_cmyka */
#ifdef GL_EXT_color_buffer_float
  _glewInfo_GL_EXT_color_buffer_float();
#endif /* GL_EXT_color_buffer_float */
#ifdef GL_EXT_color_buffer_half_float
  _glewInfo_GL_EXT_color_buffer_half_float();
#endif /* GL_EXT_color_buffer_half_float */
#ifdef GL_EXT_color_subtable
  _glewInfo_GL_EXT_color_subtable();
#endif /* GL_EXT_color_subtable */
#ifdef GL_EXT_compiled_vertex_array
  _glewInfo_GL_EXT_compiled_vertex_array();
#endif /* GL_EXT_compiled_vertex_array */
#ifdef GL_EXT_compressed_ETC1_RGB8_sub_texture
  _glewInfo_GL_EXT_compressed_ETC1_RGB8_sub_texture();
#endif /* GL_EXT_compressed_ETC1_RGB8_sub_texture */
#ifdef GL_EXT_conservative_depth
  _glewInfo_GL_EXT_conservative_depth();
#endif /* GL_EXT_conservative_depth */
#ifdef GL_EXT_convolution
  _glewInfo_GL_EXT_convolution();
#endif /* GL_EXT_convolution */
#ifdef GL_EXT_coordinate_frame
  _glewInfo_GL_EXT_coordinate_frame();
#endif /* GL_EXT_coordinate_frame */
#ifdef GL_EXT_copy_image
  _glewInfo_GL_EXT_copy_image();
#endif /* GL_EXT_copy_image */
#ifdef GL_EXT_copy_texture
  _glewInfo_GL_EXT_copy_texture();
#endif /* GL_EXT_copy_texture */
#ifdef GL_EXT_cull_vertex
  _glewInfo_GL_EXT_cull_vertex();
#endif /* GL_EXT_cull_vertex */
#ifdef GL_EXT_debug_label
  _glewInfo_GL_EXT_debug_label();
#endif /* GL_EXT_debug_label */
#ifdef GL_EXT_debug_marker
  _glewInfo_GL_EXT_debug_marker();
#endif /* GL_EXT_debug_marker */
#ifdef GL_EXT_depth_bounds_test
  _glewInfo_GL_EXT_depth_bounds_test();
#endif /* GL_EXT_depth_bounds_test */
#ifdef GL_EXT_depth_clamp
  _glewInfo_GL_EXT_depth_clamp();
#endif /* GL_EXT_depth_clamp */
#ifdef GL_EXT_direct_state_access
  _glewInfo_GL_EXT_direct_state_access();
#endif /* GL_EXT_direct_state_access */
#ifdef GL_EXT_discard_framebuffer
  _glewInfo_GL_EXT_discard_framebuffer();
#endif /* GL_EXT_discard_framebuffer */
#ifdef GL_EXT_disjoint_timer_query
  _glewInfo_GL_EXT_disjoint_timer_query();
#endif /* GL_EXT_disjoint_timer_query */
#ifdef GL_EXT_draw_buffers
  _glewInfo_GL_EXT_draw_buffers();
#endif /* GL_EXT_draw_buffers */
#ifdef GL_EXT_draw_buffers2
  _glewInfo_GL_EXT_draw_buffers2();
#endif /* GL_EXT_draw_buffers2 */
#ifdef GL_EXT_draw_buffers_indexed
  _glewInfo_GL_EXT_draw_buffers_indexed();
#endif /* GL_EXT_draw_buffers_indexed */
#ifdef GL_EXT_draw_elements_base_vertex
  _glewInfo_GL_EXT_draw_elements_base_vertex();
#endif /* GL_EXT_draw_elements_base_vertex */
#ifdef GL_EXT_draw_instanced
  _glewInfo_GL_EXT_draw_instanced();
#endif /* GL_EXT_draw_instanced */
#ifdef GL_EXT_draw_range_elements
  _glewInfo_GL_EXT_draw_range_elements();
#endif /* GL_EXT_draw_range_elements */
#ifdef GL_EXT_draw_transform_feedback
  _glewInfo_GL_EXT_draw_transform_feedback();
#endif /* GL_EXT_draw_transform_feedback */
#ifdef GL_EXT_external_buffer
  _glewInfo_GL_EXT_external_buffer();
#endif /* GL_EXT_external_buffer */
#ifdef GL_EXT_float_blend
  _glewInfo_GL_EXT_float_blend();
#endif /* GL_EXT_float_blend */
#ifdef GL_EXT_fog_coord
  _glewInfo_GL_EXT_fog_coord();
#endif /* GL_EXT_fog_coord */
#ifdef GL_EXT_frag_depth
  _glewInfo_GL_EXT_frag_depth();
#endif /* GL_EXT_frag_depth */
#ifdef GL_EXT_fragment_lighting
  _glewInfo_GL_EXT_fragment_lighting();
#endif /* GL_EXT_fragment_lighting */
#ifdef GL_EXT_framebuffer_blit
  _glewInfo_GL_EXT_framebuffer_blit();
#endif /* GL_EXT_framebuffer_blit */
#ifdef GL_EXT_framebuffer_multisample
  _glewInfo_GL_EXT_framebuffer_multisample();
#endif /* GL_EXT_framebuffer_multisample */
#ifdef GL_EXT_framebuffer_multisample_blit_scaled
  _glewInfo_GL_EXT_framebuffer_multisample_blit_scaled();
#endif /* GL_EXT_framebuffer_multisample_blit_scaled */
#ifdef GL_EXT_framebuffer_object
  _glewInfo_GL_EXT_framebuffer_object();
#endif /* GL_EXT_framebuffer_object */
#ifdef GL_EXT_framebuffer_sRGB
  _glewInfo_GL_EXT_framebuffer_sRGB();
#endif /* GL_EXT_framebuffer_sRGB */
#ifdef GL_EXT_geometry_point_size
  _glewInfo_GL_EXT_geometry_point_size();
#endif /* GL_EXT_geometry_point_size */
#ifdef GL_EXT_geometry_shader
  _glewInfo_GL_EXT_geometry_shader();
#endif /* GL_EXT_geometry_shader */
#ifdef GL_EXT_geometry_shader4
  _glewInfo_GL_EXT_geometry_shader4();
#endif /* GL_EXT_geometry_shader4 */
#ifdef GL_EXT_gpu_program_parameters
  _glewInfo_GL_EXT_gpu_program_parameters();
#endif /* GL_EXT_gpu_program_parameters */
#ifdef GL_EXT_gpu_shader4
  _glewInfo_GL_EXT_gpu_shader4();
#endif /* GL_EXT_gpu_shader4 */
#ifdef GL_EXT_gpu_shader5
  _glewInfo_GL_EXT_gpu_shader5();
#endif /* GL_EXT_gpu_shader5 */
#ifdef GL_EXT_histogram
  _glewInfo_GL_EXT_histogram();
#endif /* GL_EXT_histogram */
#ifdef GL_EXT_index_array_formats
  _glewInfo_GL_EXT_index_array_formats();
#endif /* GL_EXT_index_array_formats */
#ifdef GL_EXT_index_func
  _glewInfo_GL_EXT_index_func();
#endif /* GL_EXT_index_func */
#ifdef GL_EXT_index_material
  _glewInfo_GL_EXT_index_material();
#endif /* GL_EXT_index_material */
#ifdef GL_EXT_index_texture
  _glewInfo_GL_EXT_index_texture();
#endif /* GL_EXT_index_texture */
#ifdef GL_EXT_instanced_arrays
  _glewInfo_GL_EXT_instanced_arrays();
#endif /* GL_EXT_instanced_arrays */
#ifdef GL_EXT_light_texture
  _glewInfo_GL_EXT_light_texture();
#endif /* GL_EXT_light_texture */
#ifdef GL_EXT_map_buffer_range
  _glewInfo_GL_EXT_map_buffer_range();
#endif /* GL_EXT_map_buffer_range */
#ifdef GL_EXT_memory_object
  _glewInfo_GL_EXT_memory_object();
#endif /* GL_EXT_memory_object */
#ifdef GL_EXT_memory_object_fd
  _glewInfo_GL_EXT_memory_object_fd();
#endif /* GL_EXT_memory_object_fd */
#ifdef GL_EXT_memory_object_win32
  _glewInfo_GL_EXT_memory_object_win32();
#endif /* GL_EXT_memory_object_win32 */
#ifdef GL_EXT_misc_attribute
  _glewInfo_GL_EXT_misc_attribute();
#endif /* GL_EXT_misc_attribute */
#ifdef GL_EXT_multi_draw_arrays
  _glewInfo_GL_EXT_multi_draw_arrays();
#endif /* GL_EXT_multi_draw_arrays */
#ifdef GL_EXT_multi_draw_indirect
  _glewInfo_GL_EXT_multi_draw_indirect();
#endif /* GL_EXT_multi_draw_indirect */
#ifdef GL_EXT_multiple_textures
  _glewInfo_GL_EXT_multiple_textures();
#endif /* GL_EXT_multiple_textures */
#ifdef GL_EXT_multisample
  _glewInfo_GL_EXT_multisample();
#endif /* GL_EXT_multisample */
#ifdef GL_EXT_multisample_compatibility
  _glewInfo_GL_EXT_multisample_compatibility();
#endif /* GL_EXT_multisample_compatibility */
#ifdef GL_EXT_multisampled_render_to_texture
  _glewInfo_GL_EXT_multisampled_render_to_texture();
#endif /* GL_EXT_multisampled_render_to_texture */
#ifdef GL_EXT_multisampled_render_to_texture2
  _glewInfo_GL_EXT_multisampled_render_to_texture2();
#endif /* GL_EXT_multisampled_render_to_texture2 */
#ifdef GL_EXT_multiview_draw_buffers
  _glewInfo_GL_EXT_multiview_draw_buffers();
#endif /* GL_EXT_multiview_draw_buffers */
#ifdef GL_EXT_multiview_tessellation_geometry_shader
  _glewInfo_GL_EXT_multiview_tessellation_geometry_shader();
#endif /* GL_EXT_multiview_tessellation_geometry_shader */
#ifdef GL_EXT_multiview_texture_multisample
  _glewInfo_GL_EXT_multiview_texture_multisample();
#endif /* GL_EXT_multiview_texture_multisample */
#ifdef GL_EXT_multiview_timer_query
  _glewInfo_GL_EXT_multiview_timer_query();
#endif /* GL_EXT_multiview_timer_query */
#ifdef GL_EXT_occlusion_query_boolean
  _glewInfo_GL_EXT_occlusion_query_boolean();
#endif /* GL_EXT_occlusion_query_boolean */
#ifdef GL_EXT_packed_depth_stencil
  _glewInfo_GL_EXT_packed_depth_stencil();
#endif /* GL_EXT_packed_depth_stencil */
#ifdef GL_EXT_packed_float
  _glewInfo_GL_EXT_packed_float();
#endif /* GL_EXT_packed_float */
#ifdef GL_EXT_packed_pixels
  _glewInfo_GL_EXT_packed_pixels();
#endif /* GL_EXT_packed_pixels */
#ifdef GL_EXT_paletted_texture
  _glewInfo_GL_EXT_paletted_texture();
#endif /* GL_EXT_paletted_texture */
#ifdef GL_EXT_pixel_buffer_object
  _glewInfo_GL_EXT_pixel_buffer_object();
#endif /* GL_EXT_pixel_buffer_object */
#ifdef GL_EXT_pixel_transform
  _glewInfo_GL_EXT_pixel_transform();
#endif /* GL_EXT_pixel_transform */
#ifdef GL_EXT_pixel_transform_color_table
  _glewInfo_GL_EXT_pixel_transform_color_table();
#endif /* GL_EXT_pixel_transform_color_table */
#ifdef GL_EXT_point_parameters
  _glewInfo_GL_EXT_point_parameters();
#endif /* GL_EXT_point_parameters */
#ifdef GL_EXT_polygon_offset
  _glewInfo_GL_EXT_polygon_offset();
#endif /* GL_EXT_polygon_offset */
#ifdef GL_EXT_polygon_offset_clamp
  _glewInfo_GL_EXT_polygon_offset_clamp();
#endif /* GL_EXT_polygon_offset_clamp */
#ifdef GL_EXT_post_depth_coverage
  _glewInfo_GL_EXT_post_depth_coverage();
#endif /* GL_EXT_post_depth_coverage */
#ifdef GL_EXT_primitive_bounding_box
  _glewInfo_GL_EXT_primitive_bounding_box();
#endif /* GL_EXT_primitive_bounding_box */
#ifdef GL_EXT_protected_textures
  _glewInfo_GL_EXT_protected_textures();
#endif /* GL_EXT_protected_textures */
#ifdef GL_EXT_provoking_vertex
  _glewInfo_GL_EXT_provoking_vertex();
#endif /* GL_EXT_provoking_vertex */
#ifdef GL_EXT_pvrtc_sRGB
  _glewInfo_GL_EXT_pvrtc_sRGB();
#endif /* GL_EXT_pvrtc_sRGB */
#ifdef GL_EXT_raster_multisample
  _glewInfo_GL_EXT_raster_multisample();
#endif /* GL_EXT_raster_multisample */
#ifdef GL_EXT_read_format_bgra
  _glewInfo_GL_EXT_read_format_bgra();
#endif /* GL_EXT_read_format_bgra */
#ifdef GL_EXT_render_snorm
  _glewInfo_GL_EXT_render_snorm();
#endif /* GL_EXT_render_snorm */
#ifdef GL_EXT_rescale_normal
  _glewInfo_GL_EXT_rescale_normal();
#endif /* GL_EXT_rescale_normal */
#ifdef GL_EXT_robustness
  _glewInfo_GL_EXT_robustness();
#endif /* GL_EXT_robustness */
#ifdef GL_EXT_sRGB
  _glewInfo_GL_EXT_sRGB();
#endif /* GL_EXT_sRGB */
#ifdef GL_EXT_sRGB_write_control
  _glewInfo_GL_EXT_sRGB_write_control();
#endif /* GL_EXT_sRGB_write_control */
#ifdef GL_EXT_scene_marker
  _glewInfo_GL_EXT_scene_marker();
#endif /* GL_EXT_scene_marker */
#ifdef GL_EXT_secondary_color
  _glewInfo_GL_EXT_secondary_color();
#endif /* GL_EXT_secondary_color */
#ifdef GL_EXT_semaphore
  _glewInfo_GL_EXT_semaphore();
#endif /* GL_EXT_semaphore */
#ifdef GL_EXT_semaphore_fd
  _glewInfo_GL_EXT_semaphore_fd();
#endif /* GL_EXT_semaphore_fd */
#ifdef GL_EXT_semaphore_win32
  _glewInfo_GL_EXT_semaphore_win32();
#endif /* GL_EXT_semaphore_win32 */
#ifdef GL_EXT_separate_shader_objects
  _glewInfo_GL_EXT_separate_shader_objects();
#endif /* GL_EXT_separate_shader_objects */
#ifdef GL_EXT_separate_specular_color
  _glewInfo_GL_EXT_separate_specular_color();
#endif /* GL_EXT_separate_specular_color */
#ifdef GL_EXT_shader_framebuffer_fetch
  _glewInfo_GL_EXT_shader_framebuffer_fetch();
#endif /* GL_EXT_shader_framebuffer_fetch */
#ifdef GL_EXT_shader_framebuffer_fetch_non_coherent
  _glewInfo_GL_EXT_shader_framebuffer_fetch_non_coherent();
#endif /* GL_EXT_shader_framebuffer_fetch_non_coherent */
#ifdef GL_EXT_shader_group_vote
  _glewInfo_GL_EXT_shader_group_vote();
#endif /* GL_EXT_shader_group_vote */
#ifdef GL_EXT_shader_image_load_formatted
  _glewInfo_GL_EXT_shader_image_load_formatted();
#endif /* GL_EXT_shader_image_load_formatted */
#ifdef GL_EXT_shader_image_load_store
  _glewInfo_GL_EXT_shader_image_load_store();
#endif /* GL_EXT_shader_image_load_store */
#ifdef GL_EXT_shader_implicit_conversions
  _glewInfo_GL_EXT_shader_implicit_conversions();
#endif /* GL_EXT_shader_implicit_conversions */
#ifdef GL_EXT_shader_integer_mix
  _glewInfo_GL_EXT_shader_integer_mix();
#endif /* GL_EXT_shader_integer_mix */
#ifdef GL_EXT_shader_io_blocks
  _glewInfo_GL_EXT_shader_io_blocks();
#endif /* GL_EXT_shader_io_blocks */
#ifdef GL_EXT_shader_non_constant_global_initializers
  _glewInfo_GL_EXT_shader_non_constant_global_initializers();
#endif /* GL_EXT_shader_non_constant_global_initializers */
#ifdef GL_EXT_shader_pixel_local_storage
  _glewInfo_GL_EXT_shader_pixel_local_storage();
#endif /* GL_EXT_shader_pixel_local_storage */
#ifdef GL_EXT_shader_pixel_local_storage2
  _glewInfo_GL_EXT_shader_pixel_local_storage2();
#endif /* GL_EXT_shader_pixel_local_storage2 */
#ifdef GL_EXT_shader_texture_lod
  _glewInfo_GL_EXT_shader_texture_lod();
#endif /* GL_EXT_shader_texture_lod */
#ifdef GL_EXT_shadow_funcs
  _glewInfo_GL_EXT_shadow_funcs();
#endif /* GL_EXT_shadow_funcs */
#ifdef GL_EXT_shadow_samplers
  _glewInfo_GL_EXT_shadow_samplers();
#endif /* GL_EXT_shadow_samplers */
#ifdef GL_EXT_shared_texture_palette
  _glewInfo_GL_EXT_shared_texture_palette();
#endif /* GL_EXT_shared_texture_palette */
#ifdef GL_EXT_sparse_texture
  _glewInfo_GL_EXT_sparse_texture();
#endif /* GL_EXT_sparse_texture */
#ifdef GL_EXT_sparse_texture2
  _glewInfo_GL_EXT_sparse_texture2();
#endif /* GL_EXT_sparse_texture2 */
#ifdef GL_EXT_static_vertex_array
  _glewInfo_GL_EXT_static_vertex_array();
#endif /* GL_EXT_static_vertex_array */
#ifdef GL_EXT_stencil_clear_tag
  _glewInfo_GL_EXT_stencil_clear_tag();
#endif /* GL_EXT_stencil_clear_tag */
#ifdef GL_EXT_stencil_two_side
  _glewInfo_GL_EXT_stencil_two_side();
#endif /* GL_EXT_stencil_two_side */
#ifdef GL_EXT_stencil_wrap
  _glewInfo_GL_EXT_stencil_wrap();
#endif /* GL_EXT_stencil_wrap */
#ifdef GL_EXT_subtexture
  _glewInfo_GL_EXT_subtexture();
#endif /* GL_EXT_subtexture */
#ifdef GL_EXT_tessellation_point_size
  _glewInfo_GL_EXT_tessellation_point_size();
#endif /* GL_EXT_tessellation_point_size */
#ifdef GL_EXT_tessellation_shader
  _glewInfo_GL_EXT_tessellation_shader();
#endif /* GL_EXT_tessellation_shader */
#ifdef GL_EXT_texture
  _glewInfo_GL_EXT_texture();
#endif /* GL_EXT_texture */
#ifdef GL_EXT_texture3D
  _glewInfo_GL_EXT_texture3D();
#endif /* GL_EXT_texture3D */
#ifdef GL_EXT_texture_array
  _glewInfo_GL_EXT_texture_array();
#endif /* GL_EXT_texture_array */
#ifdef GL_EXT_texture_border_clamp
  _glewInfo_GL_EXT_texture_border_clamp();
#endif /* GL_EXT_texture_border_clamp */
#ifdef GL_EXT_texture_buffer
  _glewInfo_GL_EXT_texture_buffer();
#endif /* GL_EXT_texture_buffer */
#ifdef GL_EXT_texture_buffer_object
  _glewInfo_GL_EXT_texture_buffer_object();
#endif /* GL_EXT_texture_buffer_object */
#ifdef GL_EXT_texture_compression_astc_decode_mode
  _glewInfo_GL_EXT_texture_compression_astc_decode_mode();
#endif /* GL_EXT_texture_compression_astc_decode_mode */
#ifdef GL_EXT_texture_compression_astc_decode_mode_rgb9e5
  _glewInfo_GL_EXT_texture_compression_astc_decode_mode_rgb9e5();
#endif /* GL_EXT_texture_compression_astc_decode_mode_rgb9e5 */
#ifdef GL_EXT_texture_compression_bptc
  _glewInfo_GL_EXT_texture_compression_bptc();
#endif /* GL_EXT_texture_compression_bptc */
#ifdef GL_EXT_texture_compression_dxt1
  _glewInfo_GL_EXT_texture_compression_dxt1();
#endif /* GL_EXT_texture_compression_dxt1 */
#ifdef GL_EXT_texture_compression_latc
  _glewInfo_GL_EXT_texture_compression_latc();
#endif /* GL_EXT_texture_compression_latc */
#ifdef GL_EXT_texture_compression_rgtc
  _glewInfo_GL_EXT_texture_compression_rgtc();
#endif /* GL_EXT_texture_compression_rgtc */
#ifdef GL_EXT_texture_compression_s3tc
  _glewInfo_GL_EXT_texture_compression_s3tc();
#endif /* GL_EXT_texture_compression_s3tc */
#ifdef GL_EXT_texture_compression_s3tc_srgb
  _glewInfo_GL_EXT_texture_compression_s3tc_srgb();
#endif /* GL_EXT_texture_compression_s3tc_srgb */
#ifdef GL_EXT_texture_cube_map
  _glewInfo_GL_EXT_texture_cube_map();
#endif /* GL_EXT_texture_cube_map */
#ifdef GL_EXT_texture_cube_map_array
  _glewInfo_GL_EXT_texture_cube_map_array();
#endif /* GL_EXT_texture_cube_map_array */
#ifdef GL_EXT_texture_edge_clamp
  _glewInfo_GL_EXT_texture_edge_clamp();
#endif /* GL_EXT_texture_edge_clamp */
#ifdef GL_EXT_texture_env
  _glewInfo_GL_EXT_texture_env();
#endif /* GL_EXT_texture_env */
#ifdef GL_EXT_texture_env_add
  _glewInfo_GL_EXT_texture_env_add();
#endif /* GL_EXT_texture_env_add */
#ifdef GL_EXT_texture_env_combine
  _glewInfo_GL_EXT_texture_env_combine();
#endif /* GL_EXT_texture_env_combine */
#ifdef GL_EXT_texture_env_dot3
  _glewInfo_GL_EXT_texture_env_dot3();
#endif /* GL_EXT_texture_env_dot3 */
#ifdef GL_EXT_texture_filter_anisotropic
  _glewInfo_GL_EXT_texture_filter_anisotropic();
#endif /* GL_EXT_texture_filter_anisotropic */
#ifdef GL_EXT_texture_filter_minmax
  _glewInfo_GL_EXT_texture_filter_minmax();
#endif /* GL_EXT_texture_filter_minmax */
#ifdef GL_EXT_texture_format_BGRA8888
  _glewInfo_GL_EXT_texture_format_BGRA8888();
#endif /* GL_EXT_texture_format_BGRA8888 */
#ifdef GL_EXT_texture_format_sRGB_override
  _glewInfo_GL_EXT_texture_format_sRGB_override();
#endif /* GL_EXT_texture_format_sRGB_override */
#ifdef GL_EXT_texture_integer
  _glewInfo_GL_EXT_texture_integer();
#endif /* GL_EXT_texture_integer */
#ifdef GL_EXT_texture_lod_bias
  _glewInfo_GL_EXT_texture_lod_bias();
#endif /* GL_EXT_texture_lod_bias */
#ifdef GL_EXT_texture_mirror_clamp
  _glewInfo_GL_EXT_texture_mirror_clamp();
#endif /* GL_EXT_texture_mirror_clamp */
#ifdef GL_EXT_texture_mirror_clamp_to_edge
  _glewInfo_GL_EXT_texture_mirror_clamp_to_edge();
#endif /* GL_EXT_texture_mirror_clamp_to_edge */
#ifdef GL_EXT_texture_norm16
  _glewInfo_GL_EXT_texture_norm16();
#endif /* GL_EXT_texture_norm16 */
#ifdef GL_EXT_texture_object
  _glewInfo_GL_EXT_texture_object();
#endif /* GL_EXT_texture_object */
#ifdef GL_EXT_texture_perturb_normal
  _glewInfo_GL_EXT_texture_perturb_normal();
#endif /* GL_EXT_texture_perturb_normal */
#ifdef GL_EXT_texture_query_lod
  _glewInfo_GL_EXT_texture_query_lod();
#endif /* GL_EXT_texture_query_lod */
#ifdef GL_EXT_texture_rectangle
  _glewInfo_GL_EXT_texture_rectangle();
#endif /* GL_EXT_texture_rectangle */
#ifdef GL_EXT_texture_rg
  _glewInfo_GL_EXT_texture_rg();
#endif /* GL_EXT_texture_rg */
#ifdef GL_EXT_texture_sRGB
  _glewInfo_GL_EXT_texture_sRGB();
#endif /* GL_EXT_texture_sRGB */
#ifdef GL_EXT_texture_sRGB_R8
  _glewInfo_GL_EXT_texture_sRGB_R8();
#endif /* GL_EXT_texture_sRGB_R8 */
#ifdef GL_EXT_texture_sRGB_RG8
  _glewInfo_GL_EXT_texture_sRGB_RG8();
#endif /* GL_EXT_texture_sRGB_RG8 */
#ifdef GL_EXT_texture_sRGB_decode
  _glewInfo_GL_EXT_texture_sRGB_decode();
#endif /* GL_EXT_texture_sRGB_decode */
#ifdef GL_EXT_texture_shadow_lod
  _glewInfo_GL_EXT_texture_shadow_lod();
#endif /* GL_EXT_texture_shadow_lod */
#ifdef GL_EXT_texture_shared_exponent
  _glewInfo_GL_EXT_texture_shared_exponent();
#endif /* GL_EXT_texture_shared_exponent */
#ifdef GL_EXT_texture_snorm
  _glewInfo_GL_EXT_texture_snorm();
#endif /* GL_EXT_texture_snorm */
#ifdef GL_EXT_texture_storage
  _glewInfo_GL_EXT_texture_storage();
#endif /* GL_EXT_texture_storage */
#ifdef GL_EXT_texture_swizzle
  _glewInfo_GL_EXT_texture_swizzle();
#endif /* GL_EXT_texture_swizzle */
#ifdef GL_EXT_texture_type_2_10_10_10_REV
  _glewInfo_GL_EXT_texture_type_2_10_10_10_REV();
#endif /* GL_EXT_texture_type_2_10_10_10_REV */
#ifdef GL_EXT_texture_view
  _glewInfo_GL_EXT_texture_view();
#endif /* GL_EXT_texture_view */
#ifdef GL_EXT_timer_query
  _glewInfo_GL_EXT_timer_query();
#endif /* GL_EXT_timer_query */
#ifdef GL_EXT_transform_feedback
  _glewInfo_GL_EXT_transform_feedback();
#endif /* GL_EXT_transform_feedback */
#ifdef GL_EXT_unpack_subimage
  _glewInfo_GL_EXT_unpack_subimage();
#endif /* GL_EXT_unpack_subimage */
#ifdef GL_EXT_vertex_array
  _glewInfo_GL_EXT_vertex_array();
#endif /* GL_EXT_vertex_array */
#ifdef GL_EXT_vertex_array_bgra
  _glewInfo_GL_EXT_vertex_array_bgra();
#endif /* GL_EXT_vertex_array_bgra */
#ifdef GL_EXT_vertex_array_setXXX
  _glewInfo_GL_EXT_vertex_array_setXXX();
#endif /* GL_EXT_vertex_array_setXXX */
#ifdef GL_EXT_vertex_attrib_64bit
  _glewInfo_GL_EXT_vertex_attrib_64bit();
#endif /* GL_EXT_vertex_attrib_64bit */
#ifdef GL_EXT_vertex_shader
  _glewInfo_GL_EXT_vertex_shader();
#endif /* GL_EXT_vertex_shader */
#ifdef GL_EXT_vertex_weighting
  _glewInfo_GL_EXT_vertex_weighting();
#endif /* GL_EXT_vertex_weighting */
#ifdef GL_EXT_win32_keyed_mutex
  _glewInfo_GL_EXT_win32_keyed_mutex();
#endif /* GL_EXT_win32_keyed_mutex */
#ifdef GL_EXT_window_rectangles
  _glewInfo_GL_EXT_window_rectangles();
#endif /* GL_EXT_window_rectangles */
#ifdef GL_EXT_x11_sync_object
  _glewInfo_GL_EXT_x11_sync_object();
#endif /* GL_EXT_x11_sync_object */
#ifdef GL_FJ_shader_binary_GCCSO
  _glewInfo_GL_FJ_shader_binary_GCCSO();
#endif /* GL_FJ_shader_binary_GCCSO */
#ifdef GL_GREMEDY_frame_terminator
  _glewInfo_GL_GREMEDY_frame_terminator();
#endif /* GL_GREMEDY_frame_terminator */
#ifdef GL_GREMEDY_string_marker
  _glewInfo_GL_GREMEDY_string_marker();
#endif /* GL_GREMEDY_string_marker */
#ifdef GL_HP_convolution_border_modes
  _glewInfo_GL_HP_convolution_border_modes();
#endif /* GL_HP_convolution_border_modes */
#ifdef GL_HP_image_transform
  _glewInfo_GL_HP_image_transform();
#endif /* GL_HP_image_transform */
#ifdef GL_HP_occlusion_test
  _glewInfo_GL_HP_occlusion_test();
#endif /* GL_HP_occlusion_test */
#ifdef GL_HP_texture_lighting
  _glewInfo_GL_HP_texture_lighting();
#endif /* GL_HP_texture_lighting */
#ifdef GL_IBM_cull_vertex
  _glewInfo_GL_IBM_cull_vertex();
#endif /* GL_IBM_cull_vertex */
#ifdef GL_IBM_multimode_draw_arrays
  _glewInfo_GL_IBM_multimode_draw_arrays();
#endif /* GL_IBM_multimode_draw_arrays */
#ifdef GL_IBM_rasterpos_clip
  _glewInfo_GL_IBM_rasterpos_clip();
#endif /* GL_IBM_rasterpos_clip */
#ifdef GL_IBM_static_data
  _glewInfo_GL_IBM_static_data();
#endif /* GL_IBM_static_data */
#ifdef GL_IBM_texture_mirrored_repeat
  _glewInfo_GL_IBM_texture_mirrored_repeat();
#endif /* GL_IBM_texture_mirrored_repeat */
#ifdef GL_IBM_vertex_array_lists
  _glewInfo_GL_IBM_vertex_array_lists();
#endif /* GL_IBM_vertex_array_lists */
#ifdef GL_IMG_bindless_texture
  _glewInfo_GL_IMG_bindless_texture();
#endif /* GL_IMG_bindless_texture */
#ifdef GL_IMG_framebuffer_downsample
  _glewInfo_GL_IMG_framebuffer_downsample();
#endif /* GL_IMG_framebuffer_downsample */
#ifdef GL_IMG_multisampled_render_to_texture
  _glewInfo_GL_IMG_multisampled_render_to_texture();
#endif /* GL_IMG_multisampled_render_to_texture */
#ifdef GL_IMG_program_binary
  _glewInfo_GL_IMG_program_binary();
#endif /* GL_IMG_program_binary */
#ifdef GL_IMG_read_format
  _glewInfo_GL_IMG_read_format();
#endif /* GL_IMG_read_format */
#ifdef GL_IMG_shader_binary
  _glewInfo_GL_IMG_shader_binary();
#endif /* GL_IMG_shader_binary */
#ifdef GL_IMG_texture_compression_pvrtc
  _glewInfo_GL_IMG_texture_compression_pvrtc();
#endif /* GL_IMG_texture_compression_pvrtc */
#ifdef GL_IMG_texture_compression_pvrtc2
  _glewInfo_GL_IMG_texture_compression_pvrtc2();
#endif /* GL_IMG_texture_compression_pvrtc2 */
#ifdef GL_IMG_texture_env_enhanced_fixed_function
  _glewInfo_GL_IMG_texture_env_enhanced_fixed_function();
#endif /* GL_IMG_texture_env_enhanced_fixed_function */
#ifdef GL_IMG_texture_filter_cubic
  _glewInfo_GL_IMG_texture_filter_cubic();
#endif /* GL_IMG_texture_filter_cubic */
#ifdef GL_INGR_color_clamp
  _glewInfo_GL_INGR_color_clamp();
#endif /* GL_INGR_color_clamp */
#ifdef GL_INGR_interlace_read
  _glewInfo_GL_INGR_interlace_read();
#endif /* GL_INGR_interlace_read */
#ifdef GL_INTEL_blackhole_render
  _glewInfo_GL_INTEL_blackhole_render();
#endif /* GL_INTEL_blackhole_render */
#ifdef GL_INTEL_conservative_rasterization
  _glewInfo_GL_INTEL_conservative_rasterization();
#endif /* GL_INTEL_conservative_rasterization */
#ifdef GL_INTEL_fragment_shader_ordering
  _glewInfo_GL_INTEL_fragment_shader_ordering();
#endif /* GL_INTEL_fragment_shader_ordering */
#ifdef GL_INTEL_framebuffer_CMAA
  _glewInfo_GL_INTEL_framebuffer_CMAA();
#endif /* GL_INTEL_framebuffer_CMAA */
#ifdef GL_INTEL_map_texture
  _glewInfo_GL_INTEL_map_texture();
#endif /* GL_INTEL_map_texture */
#ifdef GL_INTEL_parallel_arrays
  _glewInfo_GL_INTEL_parallel_arrays();
#endif /* GL_INTEL_parallel_arrays */
#ifdef GL_INTEL_performance_query
  _glewInfo_GL_INTEL_performance_query();
#endif /* GL_INTEL_performance_query */
#ifdef GL_INTEL_shader_integer_functions2
  _glewInfo_GL_INTEL_shader_integer_functions2();
#endif /* GL_INTEL_shader_integer_functions2 */
#ifdef GL_INTEL_texture_scissor
  _glewInfo_GL_INTEL_texture_scissor();
#endif /* GL_INTEL_texture_scissor */
#ifdef GL_KHR_blend_equation_advanced
  _glewInfo_GL_KHR_blend_equation_advanced();
#endif /* GL_KHR_blend_equation_advanced */
#ifdef GL_KHR_blend_equation_advanced_coherent
  _glewInfo_GL_KHR_blend_equation_advanced_coherent();
#endif /* GL_KHR_blend_equation_advanced_coherent */
#ifdef GL_KHR_context_flush_control
  _glewInfo_GL_KHR_context_flush_control();
#endif /* GL_KHR_context_flush_control */
#ifdef GL_KHR_debug
  _glewInfo_GL_KHR_debug();
#endif /* GL_KHR_debug */
#ifdef GL_KHR_no_error
  _glewInfo_GL_KHR_no_error();
#endif /* GL_KHR_no_error */
#ifdef GL_KHR_parallel_shader_compile
  _glewInfo_GL_KHR_parallel_shader_compile();
#endif /* GL_KHR_parallel_shader_compile */
#ifdef GL_KHR_robust_buffer_access_behavior
  _glewInfo_GL_KHR_robust_buffer_access_behavior();
#endif /* GL_KHR_robust_buffer_access_behavior */
#ifdef GL_KHR_robustness
  _glewInfo_GL_KHR_robustness();
#endif /* GL_KHR_robustness */
#ifdef GL_KHR_shader_subgroup
  _glewInfo_GL_KHR_shader_subgroup();
#endif /* GL_KHR_shader_subgroup */
#ifdef GL_KHR_texture_compression_astc_hdr
  _glewInfo_GL_KHR_texture_compression_astc_hdr();
#endif /* GL_KHR_texture_compression_astc_hdr */
#ifdef GL_KHR_texture_compression_astc_ldr
  _glewInfo_GL_KHR_texture_compression_astc_ldr();
#endif /* GL_KHR_texture_compression_astc_ldr */
#ifdef GL_KHR_texture_compression_astc_sliced_3d
  _glewInfo_GL_KHR_texture_compression_astc_sliced_3d();
#endif /* GL_KHR_texture_compression_astc_sliced_3d */
#ifdef GL_KTX_buffer_region
  _glewInfo_GL_KTX_buffer_region();
#endif /* GL_KTX_buffer_region */
#ifdef GL_MESAX_texture_stack
  _glewInfo_GL_MESAX_texture_stack();
#endif /* GL_MESAX_texture_stack */
#ifdef GL_MESA_framebuffer_flip_y
  _glewInfo_GL_MESA_framebuffer_flip_y();
#endif /* GL_MESA_framebuffer_flip_y */
#ifdef GL_MESA_pack_invert
  _glewInfo_GL_MESA_pack_invert();
#endif /* GL_MESA_pack_invert */
#ifdef GL_MESA_program_binary_formats
  _glewInfo_GL_MESA_program_binary_formats();
#endif /* GL_MESA_program_binary_formats */
#ifdef GL_MESA_resize_buffers
  _glewInfo_GL_MESA_resize_buffers();
#endif /* GL_MESA_resize_buffers */
#ifdef GL_MESA_shader_integer_functions
  _glewInfo_GL_MESA_shader_integer_functions();
#endif /* GL_MESA_shader_integer_functions */
#ifdef GL_MESA_tile_raster_order
  _glewInfo_GL_MESA_tile_raster_order();
#endif /* GL_MESA_tile_raster_order */
#ifdef GL_MESA_window_pos
  _glewInfo_GL_MESA_window_pos();
#endif /* GL_MESA_window_pos */
#ifdef GL_MESA_ycbcr_texture
  _glewInfo_GL_MESA_ycbcr_texture();
#endif /* GL_MESA_ycbcr_texture */
#ifdef GL_NVX_blend_equation_advanced_multi_draw_buffers
  _glewInfo_GL_NVX_blend_equation_advanced_multi_draw_buffers();
#endif /* GL_NVX_blend_equation_advanced_multi_draw_buffers */
#ifdef GL_NVX_conditional_render
  _glewInfo_GL_NVX_conditional_render();
#endif /* GL_NVX_conditional_render */
#ifdef GL_NVX_gpu_memory_info
  _glewInfo_GL_NVX_gpu_memory_info();
#endif /* GL_NVX_gpu_memory_info */
#ifdef GL_NVX_gpu_multicast2
  _glewInfo_GL_NVX_gpu_multicast2();
#endif /* GL_NVX_gpu_multicast2 */
#ifdef GL_NVX_linked_gpu_multicast
  _glewInfo_GL_NVX_linked_gpu_multicast();
#endif /* GL_NVX_linked_gpu_multicast */
#ifdef GL_NVX_progress_fence
  _glewInfo_GL_NVX_progress_fence();
#endif /* GL_NVX_progress_fence */
#ifdef GL_NV_3dvision_settings
  _glewInfo_GL_NV_3dvision_settings();
#endif /* GL_NV_3dvision_settings */
#ifdef GL_NV_EGL_stream_consumer_external
  _glewInfo_GL_NV_EGL_stream_consumer_external();
#endif /* GL_NV_EGL_stream_consumer_external */
#ifdef GL_NV_alpha_to_coverage_dither_control
  _glewInfo_GL_NV_alpha_to_coverage_dither_control();
#endif /* GL_NV_alpha_to_coverage_dither_control */
#ifdef GL_NV_bgr
  _glewInfo_GL_NV_bgr();
#endif /* GL_NV_bgr */
#ifdef GL_NV_bindless_multi_draw_indirect
  _glewInfo_GL_NV_bindless_multi_draw_indirect();
#endif /* GL_NV_bindless_multi_draw_indirect */
#ifdef GL_NV_bindless_multi_draw_indirect_count
  _glewInfo_GL_NV_bindless_multi_draw_indirect_count();
#endif /* GL_NV_bindless_multi_draw_indirect_count */
#ifdef GL_NV_bindless_texture
  _glewInfo_GL_NV_bindless_texture();
#endif /* GL_NV_bindless_texture */
#ifdef GL_NV_blend_equation_advanced
  _glewInfo_GL_NV_blend_equation_advanced();
#endif /* GL_NV_blend_equation_advanced */
#ifdef GL_NV_blend_equation_advanced_coherent
  _glewInfo_GL_NV_blend_equation_advanced_coherent();
#endif /* GL_NV_blend_equation_advanced_coherent */
#ifdef GL_NV_blend_minmax_factor
  _glewInfo_GL_NV_blend_minmax_factor();
#endif /* GL_NV_blend_minmax_factor */
#ifdef GL_NV_blend_square
  _glewInfo_GL_NV_blend_square();
#endif /* GL_NV_blend_square */
#ifdef GL_NV_clip_space_w_scaling
  _glewInfo_GL_NV_clip_space_w_scaling();
#endif /* GL_NV_clip_space_w_scaling */
#ifdef GL_NV_command_list
  _glewInfo_GL_NV_command_list();
#endif /* GL_NV_command_list */
#ifdef GL_NV_compute_program5
  _glewInfo_GL_NV_compute_program5();
#endif /* GL_NV_compute_program5 */
#ifdef GL_NV_compute_shader_derivatives
  _glewInfo_GL_NV_compute_shader_derivatives();
#endif /* GL_NV_compute_shader_derivatives */
#ifdef GL_NV_conditional_render
  _glewInfo_GL_NV_conditional_render();
#endif /* GL_NV_conditional_render */
#ifdef GL_NV_conservative_raster
  _glewInfo_GL_NV_conservative_raster();
#endif /* GL_NV_conservative_raster */
#ifdef GL_NV_conservative_raster_dilate
  _glewInfo_GL_NV_conservative_raster_dilate();
#endif /* GL_NV_conservative_raster_dilate */
#ifdef GL_NV_conservative_raster_pre_snap
  _glewInfo_GL_NV_conservative_raster_pre_snap();
#endif /* GL_NV_conservative_raster_pre_snap */
#ifdef GL_NV_conservative_raster_pre_snap_triangles
  _glewInfo_GL_NV_conservative_raster_pre_snap_triangles();
#endif /* GL_NV_conservative_raster_pre_snap_triangles */
#ifdef GL_NV_conservative_raster_underestimation
  _glewInfo_GL_NV_conservative_raster_underestimation();
#endif /* GL_NV_conservative_raster_underestimation */
#ifdef GL_NV_copy_buffer
  _glewInfo_GL_NV_copy_buffer();
#endif /* GL_NV_copy_buffer */
#ifdef GL_NV_copy_depth_to_color
  _glewInfo_GL_NV_copy_depth_to_color();
#endif /* GL_NV_copy_depth_to_color */
#ifdef GL_NV_copy_image
  _glewInfo_GL_NV_copy_image();
#endif /* GL_NV_copy_image */
#ifdef GL_NV_deep_texture3D
  _glewInfo_GL_NV_deep_texture3D();
#endif /* GL_NV_deep_texture3D */
#ifdef GL_NV_depth_buffer_float
  _glewInfo_GL_NV_depth_buffer_float();
#endif /* GL_NV_depth_buffer_float */
#ifdef GL_NV_depth_clamp
  _glewInfo_GL_NV_depth_clamp();
#endif /* GL_NV_depth_clamp */
#ifdef GL_NV_depth_nonlinear
  _glewInfo_GL_NV_depth_nonlinear();
#endif /* GL_NV_depth_nonlinear */
#ifdef GL_NV_depth_range_unclamped
  _glewInfo_GL_NV_depth_range_unclamped();
#endif /* GL_NV_depth_range_unclamped */
#ifdef GL_NV_draw_buffers
  _glewInfo_GL_NV_draw_buffers();
#endif /* GL_NV_draw_buffers */
#ifdef GL_NV_draw_instanced
  _glewInfo_GL_NV_draw_instanced();
#endif /* GL_NV_draw_instanced */
#ifdef GL_NV_draw_texture
  _glewInfo_GL_NV_draw_texture();
#endif /* GL_NV_draw_texture */
#ifdef GL_NV_draw_vulkan_image
  _glewInfo_GL_NV_draw_vulkan_image();
#endif /* GL_NV_draw_vulkan_image */
#ifdef GL_NV_evaluators
  _glewInfo_GL_NV_evaluators();
#endif /* GL_NV_evaluators */
#ifdef GL_NV_explicit_attrib_location
  _glewInfo_GL_NV_explicit_attrib_location();
#endif /* GL_NV_explicit_attrib_location */
#ifdef GL_NV_explicit_multisample
  _glewInfo_GL_NV_explicit_multisample();
#endif /* GL_NV_explicit_multisample */
#ifdef GL_NV_fbo_color_attachments
  _glewInfo_GL_NV_fbo_color_attachments();
#endif /* GL_NV_fbo_color_attachments */
#ifdef GL_NV_fence
  _glewInfo_GL_NV_fence();
#endif /* GL_NV_fence */
#ifdef GL_NV_fill_rectangle
  _glewInfo_GL_NV_fill_rectangle();
#endif /* GL_NV_fill_rectangle */
#ifdef GL_NV_float_buffer
  _glewInfo_GL_NV_float_buffer();
#endif /* GL_NV_float_buffer */
#ifdef GL_NV_fog_distance
  _glewInfo_GL_NV_fog_distance();
#endif /* GL_NV_fog_distance */
#ifdef GL_NV_fragment_coverage_to_color
  _glewInfo_GL_NV_fragment_coverage_to_color();
#endif /* GL_NV_fragment_coverage_to_color */
#ifdef GL_NV_fragment_program
  _glewInfo_GL_NV_fragment_program();
#endif /* GL_NV_fragment_program */
#ifdef GL_NV_fragment_program2
  _glewInfo_GL_NV_fragment_program2();
#endif /* GL_NV_fragment_program2 */
#ifdef GL_NV_fragment_program4
  _glewInfo_GL_NV_fragment_program4();
#endif /* GL_NV_fragment_program4 */
#ifdef GL_NV_fragment_program_option
  _glewInfo_GL_NV_fragment_program_option();
#endif /* GL_NV_fragment_program_option */
#ifdef GL_NV_fragment_shader_barycentric
  _glewInfo_GL_NV_fragment_shader_barycentric();
#endif /* GL_NV_fragment_shader_barycentric */
#ifdef GL_NV_fragment_shader_interlock
  _glewInfo_GL_NV_fragment_shader_interlock();
#endif /* GL_NV_fragment_shader_interlock */
#ifdef GL_NV_framebuffer_blit
  _glewInfo_GL_NV_framebuffer_blit();
#endif /* GL_NV_framebuffer_blit */
#ifdef GL_NV_framebuffer_mixed_samples
  _glewInfo_GL_NV_framebuffer_mixed_samples();
#endif /* GL_NV_framebuffer_mixed_samples */
#ifdef GL_NV_framebuffer_multisample
  _glewInfo_GL_NV_framebuffer_multisample();
#endif /* GL_NV_framebuffer_multisample */
#ifdef GL_NV_framebuffer_multisample_coverage
  _glewInfo_GL_NV_framebuffer_multisample_coverage();
#endif /* GL_NV_framebuffer_multisample_coverage */
#ifdef GL_NV_generate_mipmap_sRGB
  _glewInfo_GL_NV_generate_mipmap_sRGB();
#endif /* GL_NV_generate_mipmap_sRGB */
#ifdef GL_NV_geometry_program4
  _glewInfo_GL_NV_geometry_program4();
#endif /* GL_NV_geometry_program4 */
#ifdef GL_NV_geometry_shader4
  _glewInfo_GL_NV_geometry_shader4();
#endif /* GL_NV_geometry_shader4 */
#ifdef GL_NV_geometry_shader_passthrough
  _glewInfo_GL_NV_geometry_shader_passthrough();
#endif /* GL_NV_geometry_shader_passthrough */
#ifdef GL_NV_gpu_multicast
  _glewInfo_GL_NV_gpu_multicast();
#endif /* GL_NV_gpu_multicast */
#ifdef GL_NV_gpu_program4
  _glewInfo_GL_NV_gpu_program4();
#endif /* GL_NV_gpu_program4 */
#ifdef GL_NV_gpu_program5
  _glewInfo_GL_NV_gpu_program5();
#endif /* GL_NV_gpu_program5 */
#ifdef GL_NV_gpu_program5_mem_extended
  _glewInfo_GL_NV_gpu_program5_mem_extended();
#endif /* GL_NV_gpu_program5_mem_extended */
#ifdef GL_NV_gpu_program_fp64
  _glewInfo_GL_NV_gpu_program_fp64();
#endif /* GL_NV_gpu_program_fp64 */
#ifdef GL_NV_gpu_shader5
  _glewInfo_GL_NV_gpu_shader5();
#endif /* GL_NV_gpu_shader5 */
#ifdef GL_NV_half_float
  _glewInfo_GL_NV_half_float();
#endif /* GL_NV_half_float */
#ifdef GL_NV_image_formats
  _glewInfo_GL_NV_image_formats();
#endif /* GL_NV_image_formats */
#ifdef GL_NV_instanced_arrays
  _glewInfo_GL_NV_instanced_arrays();
#endif /* GL_NV_instanced_arrays */
#ifdef GL_NV_internalformat_sample_query
  _glewInfo_GL_NV_internalformat_sample_query();
#endif /* GL_NV_internalformat_sample_query */
#ifdef GL_NV_light_max_exponent
  _glewInfo_GL_NV_light_max_exponent();
#endif /* GL_NV_light_max_exponent */
#ifdef GL_NV_memory_attachment
  _glewInfo_GL_NV_memory_attachment();
#endif /* GL_NV_memory_attachment */
#ifdef GL_NV_mesh_shader
  _glewInfo_GL_NV_mesh_shader();
#endif /* GL_NV_mesh_shader */
#ifdef GL_NV_multisample_coverage
  _glewInfo_GL_NV_multisample_coverage();
#endif /* GL_NV_multisample_coverage */
#ifdef GL_NV_multisample_filter_hint
  _glewInfo_GL_NV_multisample_filter_hint();
#endif /* GL_NV_multisample_filter_hint */
#ifdef GL_NV_non_square_matrices
  _glewInfo_GL_NV_non_square_matrices();
#endif /* GL_NV_non_square_matrices */
#ifdef GL_NV_occlusion_query
  _glewInfo_GL_NV_occlusion_query();
#endif /* GL_NV_occlusion_query */
#ifdef GL_NV_pack_subimage
  _glewInfo_GL_NV_pack_subimage();
#endif /* GL_NV_pack_subimage */
#ifdef GL_NV_packed_depth_stencil
  _glewInfo_GL_NV_packed_depth_stencil();
#endif /* GL_NV_packed_depth_stencil */
#ifdef GL_NV_packed_float
  _glewInfo_GL_NV_packed_float();
#endif /* GL_NV_packed_float */
#ifdef GL_NV_packed_float_linear
  _glewInfo_GL_NV_packed_float_linear();
#endif /* GL_NV_packed_float_linear */
#ifdef GL_NV_parameter_buffer_object
  _glewInfo_GL_NV_parameter_buffer_object();
#endif /* GL_NV_parameter_buffer_object */
#ifdef GL_NV_parameter_buffer_object2
  _glewInfo_GL_NV_parameter_buffer_object2();
#endif /* GL_NV_parameter_buffer_object2 */
#ifdef GL_NV_path_rendering
  _glewInfo_GL_NV_path_rendering();
#endif /* GL_NV_path_rendering */
#ifdef GL_NV_path_rendering_shared_edge
  _glewInfo_GL_NV_path_rendering_shared_edge();
#endif /* GL_NV_path_rendering_shared_edge */
#ifdef GL_NV_pixel_buffer_object
  _glewInfo_GL_NV_pixel_buffer_object();
#endif /* GL_NV_pixel_buffer_object */
#ifdef GL_NV_pixel_data_range
  _glewInfo_GL_NV_pixel_data_range();
#endif /* GL_NV_pixel_data_range */
#ifdef GL_NV_platform_binary
  _glewInfo_GL_NV_platform_binary();
#endif /* GL_NV_platform_binary */
#ifdef GL_NV_point_sprite
  _glewInfo_GL_NV_point_sprite();
#endif /* GL_NV_point_sprite */
#ifdef GL_NV_polygon_mode
  _glewInfo_GL_NV_polygon_mode();
#endif /* GL_NV_polygon_mode */
#ifdef GL_NV_present_video
  _glewInfo_GL_NV_present_video();
#endif /* GL_NV_present_video */
#ifdef GL_NV_primitive_restart
  _glewInfo_GL_NV_primitive_restart();
#endif /* GL_NV_primitive_restart */
#ifdef GL_NV_query_resource_tag
  _glewInfo_GL_NV_query_resource_tag();
#endif /* GL_NV_query_resource_tag */
#ifdef GL_NV_read_buffer
  _glewInfo_GL_NV_read_buffer();
#endif /* GL_NV_read_buffer */
#ifdef GL_NV_read_buffer_front
  _glewInfo_GL_NV_read_buffer_front();
#endif /* GL_NV_read_buffer_front */
#ifdef GL_NV_read_depth
  _glewInfo_GL_NV_read_depth();
#endif /* GL_NV_read_depth */
#ifdef GL_NV_read_depth_stencil
  _glewInfo_GL_NV_read_depth_stencil();
#endif /* GL_NV_read_depth_stencil */
#ifdef GL_NV_read_stencil
  _glewInfo_GL_NV_read_stencil();
#endif /* GL_NV_read_stencil */
#ifdef GL_NV_register_combiners
  _glewInfo_GL_NV_register_combiners();
#endif /* GL_NV_register_combiners */
#ifdef GL_NV_register_combiners2
  _glewInfo_GL_NV_register_combiners2();
#endif /* GL_NV_register_combiners2 */
#ifdef GL_NV_representative_fragment_test
  _glewInfo_GL_NV_representative_fragment_test();
#endif /* GL_NV_representative_fragment_test */
#ifdef GL_NV_robustness_video_memory_purge
  _glewInfo_GL_NV_robustness_video_memory_purge();
#endif /* GL_NV_robustness_video_memory_purge */
#ifdef GL_NV_sRGB_formats
  _glewInfo_GL_NV_sRGB_formats();
#endif /* GL_NV_sRGB_formats */
#ifdef GL_NV_sample_locations
  _glewInfo_GL_NV_sample_locations();
#endif /* GL_NV_sample_locations */
#ifdef GL_NV_sample_mask_override_coverage
  _glewInfo_GL_NV_sample_mask_override_coverage();
#endif /* GL_NV_sample_mask_override_coverage */
#ifdef GL_NV_scissor_exclusive
  _glewInfo_GL_NV_scissor_exclusive();
#endif /* GL_NV_scissor_exclusive */
#ifdef GL_NV_shader_atomic_counters
  _glewInfo_GL_NV_shader_atomic_counters();
#endif /* GL_NV_shader_atomic_counters */
#ifdef GL_NV_shader_atomic_float
  _glewInfo_GL_NV_shader_atomic_float();
#endif /* GL_NV_shader_atomic_float */
#ifdef GL_NV_shader_atomic_float64
  _glewInfo_GL_NV_shader_atomic_float64();
#endif /* GL_NV_shader_atomic_float64 */
#ifdef GL_NV_shader_atomic_fp16_vector
  _glewInfo_GL_NV_shader_atomic_fp16_vector();
#endif /* GL_NV_shader_atomic_fp16_vector */
#ifdef GL_NV_shader_atomic_int64
  _glewInfo_GL_NV_shader_atomic_int64();
#endif /* GL_NV_shader_atomic_int64 */
#ifdef GL_NV_shader_buffer_load
  _glewInfo_GL_NV_shader_buffer_load();
#endif /* GL_NV_shader_buffer_load */
#ifdef GL_NV_shader_noperspective_interpolation
  _glewInfo_GL_NV_shader_noperspective_interpolation();
#endif /* GL_NV_shader_noperspective_interpolation */
#ifdef GL_NV_shader_storage_buffer_object
  _glewInfo_GL_NV_shader_storage_buffer_object();
#endif /* GL_NV_shader_storage_buffer_object */
#ifdef GL_NV_shader_subgroup_partitioned
  _glewInfo_GL_NV_shader_subgroup_partitioned();
#endif /* GL_NV_shader_subgroup_partitioned */
#ifdef GL_NV_shader_texture_footprint
  _glewInfo_GL_NV_shader_texture_footprint();
#endif /* GL_NV_shader_texture_footprint */
#ifdef GL_NV_shader_thread_group
  _glewInfo_GL_NV_shader_thread_group();
#endif /* GL_NV_shader_thread_group */
#ifdef GL_NV_shader_thread_shuffle
  _glewInfo_GL_NV_shader_thread_shuffle();
#endif /* GL_NV_shader_thread_shuffle */
#ifdef GL_NV_shading_rate_image
  _glewInfo_GL_NV_shading_rate_image();
#endif /* GL_NV_shading_rate_image */
#ifdef GL_NV_shadow_samplers_array
  _glewInfo_GL_NV_shadow_samplers_array();
#endif /* GL_NV_shadow_samplers_array */
#ifdef GL_NV_shadow_samplers_cube
  _glewInfo_GL_NV_shadow_samplers_cube();
#endif /* GL_NV_shadow_samplers_cube */
#ifdef GL_NV_stereo_view_rendering
  _glewInfo_GL_NV_stereo_view_rendering();
#endif /* GL_NV_stereo_view_rendering */
#ifdef GL_NV_tessellation_program5
  _glewInfo_GL_NV_tessellation_program5();
#endif /* GL_NV_tessellation_program5 */
#ifdef GL_NV_texgen_emboss
  _glewInfo_GL_NV_texgen_emboss();
#endif /* GL_NV_texgen_emboss */
#ifdef GL_NV_texgen_reflection
  _glewInfo_GL_NV_texgen_reflection();
#endif /* GL_NV_texgen_reflection */
#ifdef GL_NV_texture_array
  _glewInfo_GL_NV_texture_array();
#endif /* GL_NV_texture_array */
#ifdef GL_NV_texture_barrier
  _glewInfo_GL_NV_texture_barrier();
#endif /* GL_NV_texture_barrier */
#ifdef GL_NV_texture_border_clamp
  _glewInfo_GL_NV_texture_border_clamp();
#endif /* GL_NV_texture_border_clamp */
#ifdef GL_NV_texture_compression_latc
  _glewInfo_GL_NV_texture_compression_latc();
#endif /* GL_NV_texture_compression_latc */
#ifdef GL_NV_texture_compression_s3tc
  _glewInfo_GL_NV_texture_compression_s3tc();
#endif /* GL_NV_texture_compression_s3tc */
#ifdef GL_NV_texture_compression_s3tc_update
  _glewInfo_GL_NV_texture_compression_s3tc_update();
#endif /* GL_NV_texture_compression_s3tc_update */
#ifdef GL_NV_texture_compression_vtc
  _glewInfo_GL_NV_texture_compression_vtc();
#endif /* GL_NV_texture_compression_vtc */
#ifdef GL_NV_texture_env_combine4
  _glewInfo_GL_NV_texture_env_combine4();
#endif /* GL_NV_texture_env_combine4 */
#ifdef GL_NV_texture_expand_normal
  _glewInfo_GL_NV_texture_expand_normal();
#endif /* GL_NV_texture_expand_normal */
#ifdef GL_NV_texture_multisample
  _glewInfo_GL_NV_texture_multisample();
#endif /* GL_NV_texture_multisample */
#ifdef GL_NV_texture_npot_2D_mipmap
  _glewInfo_GL_NV_texture_npot_2D_mipmap();
#endif /* GL_NV_texture_npot_2D_mipmap */
#ifdef GL_NV_texture_rectangle
  _glewInfo_GL_NV_texture_rectangle();
#endif /* GL_NV_texture_rectangle */
#ifdef GL_NV_texture_rectangle_compressed
  _glewInfo_GL_NV_texture_rectangle_compressed();
#endif /* GL_NV_texture_rectangle_compressed */
#ifdef GL_NV_texture_shader
  _glewInfo_GL_NV_texture_shader();
#endif /* GL_NV_texture_shader */
#ifdef GL_NV_texture_shader2
  _glewInfo_GL_NV_texture_shader2();
#endif /* GL_NV_texture_shader2 */
#ifdef GL_NV_texture_shader3
  _glewInfo_GL_NV_texture_shader3();
#endif /* GL_NV_texture_shader3 */
#ifdef GL_NV_transform_feedback
  _glewInfo_GL_NV_transform_feedback();
#endif /* GL_NV_transform_feedback */
#ifdef GL_NV_transform_feedback2
  _glewInfo_GL_NV_transform_feedback2();
#endif /* GL_NV_transform_feedback2 */
#ifdef GL_NV_uniform_buffer_unified_memory
  _glewInfo_GL_NV_uniform_buffer_unified_memory();
#endif /* GL_NV_uniform_buffer_unified_memory */
#ifdef GL_NV_vdpau_interop
  _glewInfo_GL_NV_vdpau_interop();
#endif /* GL_NV_vdpau_interop */
#ifdef GL_NV_vdpau_interop2
  _glewInfo_GL_NV_vdpau_interop2();
#endif /* GL_NV_vdpau_interop2 */
#ifdef GL_NV_vertex_array_range
  _glewInfo_GL_NV_vertex_array_range();
#endif /* GL_NV_vertex_array_range */
#ifdef GL_NV_vertex_array_range2
  _glewInfo_GL_NV_vertex_array_range2();
#endif /* GL_NV_vertex_array_range2 */
#ifdef GL_NV_vertex_attrib_integer_64bit
  _glewInfo_GL_NV_vertex_attrib_integer_64bit();
#endif /* GL_NV_vertex_attrib_integer_64bit */
#ifdef GL_NV_vertex_buffer_unified_memory
  _glewInfo_GL_NV_vertex_buffer_unified_memory();
#endif /* GL_NV_vertex_buffer_unified_memory */
#ifdef GL_NV_vertex_program
  _glewInfo_GL_NV_vertex_program();
#endif /* GL_NV_vertex_program */
#ifdef GL_NV_vertex_program1_1
  _glewInfo_GL_NV_vertex_program1_1();
#endif /* GL_NV_vertex_program1_1 */
#ifdef GL_NV_vertex_program2
  _glewInfo_GL_NV_vertex_program2();
#endif /* GL_NV_vertex_program2 */
#ifdef GL_NV_vertex_program2_option
  _glewInfo_GL_NV_vertex_program2_option();
#endif /* GL_NV_vertex_program2_option */
#ifdef GL_NV_vertex_program3
  _glewInfo_GL_NV_vertex_program3();
#endif /* GL_NV_vertex_program3 */
#ifdef GL_NV_vertex_program4
  _glewInfo_GL_NV_vertex_program4();
#endif /* GL_NV_vertex_program4 */
#ifdef GL_NV_video_capture
  _glewInfo_GL_NV_video_capture();
#endif /* GL_NV_video_capture */
#ifdef GL_NV_viewport_array
  _glewInfo_GL_NV_viewport_array();
#endif /* GL_NV_viewport_array */
#ifdef GL_NV_viewport_array2
  _glewInfo_GL_NV_viewport_array2();
#endif /* GL_NV_viewport_array2 */
#ifdef GL_NV_viewport_swizzle
  _glewInfo_GL_NV_viewport_swizzle();
#endif /* GL_NV_viewport_swizzle */
#ifdef GL_OES_EGL_image
  _glewInfo_GL_OES_EGL_image();
#endif /* GL_OES_EGL_image */
#ifdef GL_OES_EGL_image_external
  _glewInfo_GL_OES_EGL_image_external();
#endif /* GL_OES_EGL_image_external */
#ifdef GL_OES_EGL_image_external_essl3
  _glewInfo_GL_OES_EGL_image_external_essl3();
#endif /* GL_OES_EGL_image_external_essl3 */
#ifdef GL_OES_blend_equation_separate
  _glewInfo_GL_OES_blend_equation_separate();
#endif /* GL_OES_blend_equation_separate */
#ifdef GL_OES_blend_func_separate
  _glewInfo_GL_OES_blend_func_separate();
#endif /* GL_OES_blend_func_separate */
#ifdef GL_OES_blend_subtract
  _glewInfo_GL_OES_blend_subtract();
#endif /* GL_OES_blend_subtract */
#ifdef GL_OES_byte_coordinates
  _glewInfo_GL_OES_byte_coordinates();
#endif /* GL_OES_byte_coordinates */
#ifdef GL_OES_compressed_ETC1_RGB8_texture
  _glewInfo_GL_OES_compressed_ETC1_RGB8_texture();
#endif /* GL_OES_compressed_ETC1_RGB8_texture */
#ifdef GL_OES_compressed_paletted_texture
  _glewInfo_GL_OES_compressed_paletted_texture();
#endif /* GL_OES_compressed_paletted_texture */
#ifdef GL_OES_copy_image
  _glewInfo_GL_OES_copy_image();
#endif /* GL_OES_copy_image */
#ifdef GL_OES_depth24
  _glewInfo_GL_OES_depth24();
#endif /* GL_OES_depth24 */
#ifdef GL_OES_depth32
  _glewInfo_GL_OES_depth32();
#endif /* GL_OES_depth32 */
#ifdef GL_OES_depth_texture
  _glewInfo_GL_OES_depth_texture();
#endif /* GL_OES_depth_texture */
#ifdef GL_OES_depth_texture_cube_map
  _glewInfo_GL_OES_depth_texture_cube_map();
#endif /* GL_OES_depth_texture_cube_map */
#ifdef GL_OES_draw_buffers_indexed
  _glewInfo_GL_OES_draw_buffers_indexed();
#endif /* GL_OES_draw_buffers_indexed */
#ifdef GL_OES_draw_texture
  _glewInfo_GL_OES_draw_texture();
#endif /* GL_OES_draw_texture */
#ifdef GL_OES_element_index_uint
  _glewInfo_GL_OES_element_index_uint();
#endif /* GL_OES_element_index_uint */
#ifdef GL_OES_extended_matrix_palette
  _glewInfo_GL_OES_extended_matrix_palette();
#endif /* GL_OES_extended_matrix_palette */
#ifdef GL_OES_fbo_render_mipmap
  _glewInfo_GL_OES_fbo_render_mipmap();
#endif /* GL_OES_fbo_render_mipmap */
#ifdef GL_OES_fragment_precision_high
  _glewInfo_GL_OES_fragment_precision_high();
#endif /* GL_OES_fragment_precision_high */
#ifdef GL_OES_framebuffer_object
  _glewInfo_GL_OES_framebuffer_object();
#endif /* GL_OES_framebuffer_object */
#ifdef GL_OES_geometry_point_size
  _glewInfo_GL_OES_geometry_point_size();
#endif /* GL_OES_geometry_point_size */
#ifdef GL_OES_geometry_shader
  _glewInfo_GL_OES_geometry_shader();
#endif /* GL_OES_geometry_shader */
#ifdef GL_OES_get_program_binary
  _glewInfo_GL_OES_get_program_binary();
#endif /* GL_OES_get_program_binary */
#ifdef GL_OES_gpu_shader5
  _glewInfo_GL_OES_gpu_shader5();
#endif /* GL_OES_gpu_shader5 */
#ifdef GL_OES_mapbuffer
  _glewInfo_GL_OES_mapbuffer();
#endif /* GL_OES_mapbuffer */
#ifdef GL_OES_matrix_get
  _glewInfo_GL_OES_matrix_get();
#endif /* GL_OES_matrix_get */
#ifdef GL_OES_matrix_palette
  _glewInfo_GL_OES_matrix_palette();
#endif /* GL_OES_matrix_palette */
#ifdef GL_OES_packed_depth_stencil
  _glewInfo_GL_OES_packed_depth_stencil();
#endif /* GL_OES_packed_depth_stencil */
#ifdef GL_OES_point_size_array
  _glewInfo_GL_OES_point_size_array();
#endif /* GL_OES_point_size_array */
#ifdef GL_OES_point_sprite
  _glewInfo_GL_OES_point_sprite();
#endif /* GL_OES_point_sprite */
#ifdef GL_OES_read_format
  _glewInfo_GL_OES_read_format();
#endif /* GL_OES_read_format */
#ifdef GL_OES_required_internalformat
  _glewInfo_GL_OES_required_internalformat();
#endif /* GL_OES_required_internalformat */
#ifdef GL_OES_rgb8_rgba8
  _glewInfo_GL_OES_rgb8_rgba8();
#endif /* GL_OES_rgb8_rgba8 */
#ifdef GL_OES_sample_shading
  _glewInfo_GL_OES_sample_shading();
#endif /* GL_OES_sample_shading */
#ifdef GL_OES_sample_variables
  _glewInfo_GL_OES_sample_variables();
#endif /* GL_OES_sample_variables */
#ifdef GL_OES_shader_image_atomic
  _glewInfo_GL_OES_shader_image_atomic();
#endif /* GL_OES_shader_image_atomic */
#ifdef GL_OES_shader_io_blocks
  _glewInfo_GL_OES_shader_io_blocks();
#endif /* GL_OES_shader_io_blocks */
#ifdef GL_OES_shader_multisample_interpolation
  _glewInfo_GL_OES_shader_multisample_interpolation();
#endif /* GL_OES_shader_multisample_interpolation */
#ifdef GL_OES_single_precision
  _glewInfo_GL_OES_single_precision();
#endif /* GL_OES_single_precision */
#ifdef GL_OES_standard_derivatives
  _glewInfo_GL_OES_standard_derivatives();
#endif /* GL_OES_standard_derivatives */
#ifdef GL_OES_stencil1
  _glewInfo_GL_OES_stencil1();
#endif /* GL_OES_stencil1 */
#ifdef GL_OES_stencil4
  _glewInfo_GL_OES_stencil4();
#endif /* GL_OES_stencil4 */
#ifdef GL_OES_stencil8
  _glewInfo_GL_OES_stencil8();
#endif /* GL_OES_stencil8 */
#ifdef GL_OES_surfaceless_context
  _glewInfo_GL_OES_surfaceless_context();
#endif /* GL_OES_surfaceless_context */
#ifdef GL_OES_tessellation_point_size
  _glewInfo_GL_OES_tessellation_point_size();
#endif /* GL_OES_tessellation_point_size */
#ifdef GL_OES_tessellation_shader
  _glewInfo_GL_OES_tessellation_shader();
#endif /* GL_OES_tessellation_shader */
#ifdef GL_OES_texture_3D
  _glewInfo_GL_OES_texture_3D();
#endif /* GL_OES_texture_3D */
#ifdef GL_OES_texture_border_clamp
  _glewInfo_GL_OES_texture_border_clamp();
#endif /* GL_OES_texture_border_clamp */
#ifdef GL_OES_texture_buffer
  _glewInfo_GL_OES_texture_buffer();
#endif /* GL_OES_texture_buffer */
#ifdef GL_OES_texture_compression_astc
  _glewInfo_GL_OES_texture_compression_astc();
#endif /* GL_OES_texture_compression_astc */
#ifdef GL_OES_texture_cube_map
  _glewInfo_GL_OES_texture_cube_map();
#endif /* GL_OES_texture_cube_map */
#ifdef GL_OES_texture_cube_map_array
  _glewInfo_GL_OES_texture_cube_map_array();
#endif /* GL_OES_texture_cube_map_array */
#ifdef GL_OES_texture_env_crossbar
  _glewInfo_GL_OES_texture_env_crossbar();
#endif /* GL_OES_texture_env_crossbar */
#ifdef GL_OES_texture_mirrored_repeat
  _glewInfo_GL_OES_texture_mirrored_repeat();
#endif /* GL_OES_texture_mirrored_repeat */
#ifdef GL_OES_texture_npot
  _glewInfo_GL_OES_texture_npot();
#endif /* GL_OES_texture_npot */
#ifdef GL_OES_texture_stencil8
  _glewInfo_GL_OES_texture_stencil8();
#endif /* GL_OES_texture_stencil8 */
#ifdef GL_OES_texture_storage_multisample_2d_array
  _glewInfo_GL_OES_texture_storage_multisample_2d_array();
#endif /* GL_OES_texture_storage_multisample_2d_array */
#ifdef GL_OES_texture_view
  _glewInfo_GL_OES_texture_view();
#endif /* GL_OES_texture_view */
#ifdef GL_OES_vertex_array_object
  _glewInfo_GL_OES_vertex_array_object();
#endif /* GL_OES_vertex_array_object */
#ifdef GL_OES_vertex_half_float
  _glewInfo_GL_OES_vertex_half_float();
#endif /* GL_OES_vertex_half_float */
#ifdef GL_OES_vertex_type_10_10_10_2
  _glewInfo_GL_OES_vertex_type_10_10_10_2();
#endif /* GL_OES_vertex_type_10_10_10_2 */
#ifdef GL_OML_interlace
  _glewInfo_GL_OML_interlace();
#endif /* GL_OML_interlace */
#ifdef GL_OML_resample
  _glewInfo_GL_OML_resample();
#endif /* GL_OML_resample */
#ifdef GL_OML_subsample
  _glewInfo_GL_OML_subsample();
#endif /* GL_OML_subsample */
#ifdef GL_OVR_multiview
  _glewInfo_GL_OVR_multiview();
#endif /* GL_OVR_multiview */
#ifdef GL_OVR_multiview2
  _glewInfo_GL_OVR_multiview2();
#endif /* GL_OVR_multiview2 */
#ifdef GL_OVR_multiview_multisampled_render_to_texture
  _glewInfo_GL_OVR_multiview_multisampled_render_to_texture();
#endif /* GL_OVR_multiview_multisampled_render_to_texture */
#ifdef GL_PGI_misc_hints
  _glewInfo_GL_PGI_misc_hints();
#endif /* GL_PGI_misc_hints */
#ifdef GL_PGI_vertex_hints
  _glewInfo_GL_PGI_vertex_hints();
#endif /* GL_PGI_vertex_hints */
#ifdef GL_QCOM_YUV_texture_gather
  _glewInfo_GL_QCOM_YUV_texture_gather();
#endif /* GL_QCOM_YUV_texture_gather */
#ifdef GL_QCOM_alpha_test
  _glewInfo_GL_QCOM_alpha_test();
#endif /* GL_QCOM_alpha_test */
#ifdef GL_QCOM_binning_control
  _glewInfo_GL_QCOM_binning_control();
#endif /* GL_QCOM_binning_control */
#ifdef GL_QCOM_driver_control
  _glewInfo_GL_QCOM_driver_control();
#endif /* GL_QCOM_driver_control */
#ifdef GL_QCOM_extended_get
  _glewInfo_GL_QCOM_extended_get();
#endif /* GL_QCOM_extended_get */
#ifdef GL_QCOM_extended_get2
  _glewInfo_GL_QCOM_extended_get2();
#endif /* GL_QCOM_extended_get2 */
#ifdef GL_QCOM_framebuffer_foveated
  _glewInfo_GL_QCOM_framebuffer_foveated();
#endif /* GL_QCOM_framebuffer_foveated */
#ifdef GL_QCOM_perfmon_global_mode
  _glewInfo_GL_QCOM_perfmon_global_mode();
#endif /* GL_QCOM_perfmon_global_mode */
#ifdef GL_QCOM_shader_framebuffer_fetch_noncoherent
  _glewInfo_GL_QCOM_shader_framebuffer_fetch_noncoherent();
#endif /* GL_QCOM_shader_framebuffer_fetch_noncoherent */
#ifdef GL_QCOM_shader_framebuffer_fetch_rate
  _glewInfo_GL_QCOM_shader_framebuffer_fetch_rate();
#endif /* GL_QCOM_shader_framebuffer_fetch_rate */
#ifdef GL_QCOM_texture_foveated
  _glewInfo_GL_QCOM_texture_foveated();
#endif /* GL_QCOM_texture_foveated */
#ifdef GL_QCOM_texture_foveated_subsampled_layout
  _glewInfo_GL_QCOM_texture_foveated_subsampled_layout();
#endif /* GL_QCOM_texture_foveated_subsampled_layout */
#ifdef GL_QCOM_tiled_rendering
  _glewInfo_GL_QCOM_tiled_rendering();
#endif /* GL_QCOM_tiled_rendering */
#ifdef GL_QCOM_writeonly_rendering
  _glewInfo_GL_QCOM_writeonly_rendering();
#endif /* GL_QCOM_writeonly_rendering */
#ifdef GL_REGAL_ES1_0_compatibility
  _glewInfo_GL_REGAL_ES1_0_compatibility();
#endif /* GL_REGAL_ES1_0_compatibility */
#ifdef GL_REGAL_ES1_1_compatibility
  _glewInfo_GL_REGAL_ES1_1_compatibility();
#endif /* GL_REGAL_ES1_1_compatibility */
#ifdef GL_REGAL_enable
  _glewInfo_GL_REGAL_enable();
#endif /* GL_REGAL_enable */
#ifdef GL_REGAL_error_string
  _glewInfo_GL_REGAL_error_string();
#endif /* GL_REGAL_error_string */
#ifdef GL_REGAL_extension_query
  _glewInfo_GL_REGAL_extension_query();
#endif /* GL_REGAL_extension_query */
#ifdef GL_REGAL_log
  _glewInfo_GL_REGAL_log();
#endif /* GL_REGAL_log */
#ifdef GL_REGAL_proc_address
  _glewInfo_GL_REGAL_proc_address();
#endif /* GL_REGAL_proc_address */
#ifdef GL_REND_screen_coordinates
  _glewInfo_GL_REND_screen_coordinates();
#endif /* GL_REND_screen_coordinates */
#ifdef GL_S3_s3tc
  _glewInfo_GL_S3_s3tc();
#endif /* GL_S3_s3tc */
#ifdef GL_SGIS_clip_band_hint
  _glewInfo_GL_SGIS_clip_band_hint();
#endif /* GL_SGIS_clip_band_hint */
#ifdef GL_SGIS_color_range
  _glewInfo_GL_SGIS_color_range();
#endif /* GL_SGIS_color_range */
#ifdef GL_SGIS_detail_texture
  _glewInfo_GL_SGIS_detail_texture();
#endif /* GL_SGIS_detail_texture */
#ifdef GL_SGIS_fog_function
  _glewInfo_GL_SGIS_fog_function();
#endif /* GL_SGIS_fog_function */
#ifdef GL_SGIS_generate_mipmap
  _glewInfo_GL_SGIS_generate_mipmap();
#endif /* GL_SGIS_generate_mipmap */
#ifdef GL_SGIS_line_texgen
  _glewInfo_GL_SGIS_line_texgen();
#endif /* GL_SGIS_line_texgen */
#ifdef GL_SGIS_multisample
  _glewInfo_GL_SGIS_multisample();
#endif /* GL_SGIS_multisample */
#ifdef GL_SGIS_multitexture
  _glewInfo_GL_SGIS_multitexture();
#endif /* GL_SGIS_multitexture */
#ifdef GL_SGIS_pixel_texture
  _glewInfo_GL_SGIS_pixel_texture();
#endif /* GL_SGIS_pixel_texture */
#ifdef GL_SGIS_point_line_texgen
  _glewInfo_GL_SGIS_point_line_texgen();
#endif /* GL_SGIS_point_line_texgen */
#ifdef GL_SGIS_shared_multisample
  _glewInfo_GL_SGIS_shared_multisample();
#endif /* GL_SGIS_shared_multisample */
#ifdef GL_SGIS_sharpen_texture
  _glewInfo_GL_SGIS_sharpen_texture();
#endif /* GL_SGIS_sharpen_texture */
#ifdef GL_SGIS_texture4D
  _glewInfo_GL_SGIS_texture4D();
#endif /* GL_SGIS_texture4D */
#ifdef GL_SGIS_texture_border_clamp
  _glewInfo_GL_SGIS_texture_border_clamp();
#endif /* GL_SGIS_texture_border_clamp */
#ifdef GL_SGIS_texture_edge_clamp
  _glewInfo_GL_SGIS_texture_edge_clamp();
#endif /* GL_SGIS_texture_edge_clamp */
#ifdef GL_SGIS_texture_filter4
  _glewInfo_GL_SGIS_texture_filter4();
#endif /* GL_SGIS_texture_filter4 */
#ifdef GL_SGIS_texture_lod
  _glewInfo_GL_SGIS_texture_lod();
#endif /* GL_SGIS_texture_lod */
#ifdef GL_SGIS_texture_select
  _glewInfo_GL_SGIS_texture_select();
#endif /* GL_SGIS_texture_select */
#ifdef GL_SGIX_async
  _glewInfo_GL_SGIX_async();
#endif /* GL_SGIX_async */
#ifdef GL_SGIX_async_histogram
  _glewInfo_GL_SGIX_async_histogram();
#endif /* GL_SGIX_async_histogram */
#ifdef GL_SGIX_async_pixel
  _glewInfo_GL_SGIX_async_pixel();
#endif /* GL_SGIX_async_pixel */
#ifdef GL_SGIX_bali_g_instruments
  _glewInfo_GL_SGIX_bali_g_instruments();
#endif /* GL_SGIX_bali_g_instruments */
#ifdef GL_SGIX_bali_r_instruments
  _glewInfo_GL_SGIX_bali_r_instruments();
#endif /* GL_SGIX_bali_r_instruments */
#ifdef GL_SGIX_bali_timer_instruments
  _glewInfo_GL_SGIX_bali_timer_instruments();
#endif /* GL_SGIX_bali_timer_instruments */
#ifdef GL_SGIX_blend_alpha_minmax
  _glewInfo_GL_SGIX_blend_alpha_minmax();
#endif /* GL_SGIX_blend_alpha_minmax */
#ifdef GL_SGIX_blend_cadd
  _glewInfo_GL_SGIX_blend_cadd();
#endif /* GL_SGIX_blend_cadd */
#ifdef GL_SGIX_blend_cmultiply
  _glewInfo_GL_SGIX_blend_cmultiply();
#endif /* GL_SGIX_blend_cmultiply */
#ifdef GL_SGIX_calligraphic_fragment
  _glewInfo_GL_SGIX_calligraphic_fragment();
#endif /* GL_SGIX_calligraphic_fragment */
#ifdef GL_SGIX_clipmap
  _glewInfo_GL_SGIX_clipmap();
#endif /* GL_SGIX_clipmap */
#ifdef GL_SGIX_color_matrix_accuracy
  _glewInfo_GL_SGIX_color_matrix_accuracy();
#endif /* GL_SGIX_color_matrix_accuracy */
#ifdef GL_SGIX_color_table_index_mode
  _glewInfo_GL_SGIX_color_table_index_mode();
#endif /* GL_SGIX_color_table_index_mode */
#ifdef GL_SGIX_complex_polar
  _glewInfo_GL_SGIX_complex_polar();
#endif /* GL_SGIX_complex_polar */
#ifdef GL_SGIX_convolution_accuracy
  _glewInfo_GL_SGIX_convolution_accuracy();
#endif /* GL_SGIX_convolution_accuracy */
#ifdef GL_SGIX_cube_map
  _glewInfo_GL_SGIX_cube_map();
#endif /* GL_SGIX_cube_map */
#ifdef GL_SGIX_cylinder_texgen
  _glewInfo_GL_SGIX_cylinder_texgen();
#endif /* GL_SGIX_cylinder_texgen */
#ifdef GL_SGIX_datapipe
  _glewInfo_GL_SGIX_datapipe();
#endif /* GL_SGIX_datapipe */
#ifdef GL_SGIX_decimation
  _glewInfo_GL_SGIX_decimation();
#endif /* GL_SGIX_decimation */
#ifdef GL_SGIX_depth_pass_instrument
  _glewInfo_GL_SGIX_depth_pass_instrument();
#endif /* GL_SGIX_depth_pass_instrument */
#ifdef GL_SGIX_depth_texture
  _glewInfo_GL_SGIX_depth_texture();
#endif /* GL_SGIX_depth_texture */
#ifdef GL_SGIX_dvc
  _glewInfo_GL_SGIX_dvc();
#endif /* GL_SGIX_dvc */
#ifdef GL_SGIX_flush_raster
  _glewInfo_GL_SGIX_flush_raster();
#endif /* GL_SGIX_flush_raster */
#ifdef GL_SGIX_fog_blend
  _glewInfo_GL_SGIX_fog_blend();
#endif /* GL_SGIX_fog_blend */
#ifdef GL_SGIX_fog_factor_to_alpha
  _glewInfo_GL_SGIX_fog_factor_to_alpha();
#endif /* GL_SGIX_fog_factor_to_alpha */
#ifdef GL_SGIX_fog_layers
  _glewInfo_GL_SGIX_fog_layers();
#endif /* GL_SGIX_fog_layers */
#ifdef GL_SGIX_fog_offset
  _glewInfo_GL_SGIX_fog_offset();
#endif /* GL_SGIX_fog_offset */
#ifdef GL_SGIX_fog_patchy
  _glewInfo_GL_SGIX_fog_patchy();
#endif /* GL_SGIX_fog_patchy */
#ifdef GL_SGIX_fog_scale
  _glewInfo_GL_SGIX_fog_scale();
#endif /* GL_SGIX_fog_scale */
#ifdef GL_SGIX_fog_texture
  _glewInfo_GL_SGIX_fog_texture();
#endif /* GL_SGIX_fog_texture */
#ifdef GL_SGIX_fragment_lighting_space
  _glewInfo_GL_SGIX_fragment_lighting_space();
#endif /* GL_SGIX_fragment_lighting_space */
#ifdef GL_SGIX_fragment_specular_lighting
  _glewInfo_GL_SGIX_fragment_specular_lighting();
#endif /* GL_SGIX_fragment_specular_lighting */
#ifdef GL_SGIX_fragments_instrument
  _glewInfo_GL_SGIX_fragments_instrument();
#endif /* GL_SGIX_fragments_instrument */
#ifdef GL_SGIX_framezoom
  _glewInfo_GL_SGIX_framezoom();
#endif /* GL_SGIX_framezoom */
#ifdef GL_SGIX_icc_texture
  _glewInfo_GL_SGIX_icc_texture();
#endif /* GL_SGIX_icc_texture */
#ifdef GL_SGIX_igloo_interface
  _glewInfo_GL_SGIX_igloo_interface();
#endif /* GL_SGIX_igloo_interface */
#ifdef GL_SGIX_image_compression
  _glewInfo_GL_SGIX_image_compression();
#endif /* GL_SGIX_image_compression */
#ifdef GL_SGIX_impact_pixel_texture
  _glewInfo_GL_SGIX_impact_pixel_texture();
#endif /* GL_SGIX_impact_pixel_texture */
#ifdef GL_SGIX_instrument_error
  _glewInfo_GL_SGIX_instrument_error();
#endif /* GL_SGIX_instrument_error */
#ifdef GL_SGIX_interlace
  _glewInfo_GL_SGIX_interlace();
#endif /* GL_SGIX_interlace */
#ifdef GL_SGIX_ir_instrument1
  _glewInfo_GL_SGIX_ir_instrument1();
#endif /* GL_SGIX_ir_instrument1 */
#ifdef GL_SGIX_line_quality_hint
  _glewInfo_GL_SGIX_line_quality_hint();
#endif /* GL_SGIX_line_quality_hint */
#ifdef GL_SGIX_list_priority
  _glewInfo_GL_SGIX_list_priority();
#endif /* GL_SGIX_list_priority */
#ifdef GL_SGIX_mpeg1
  _glewInfo_GL_SGIX_mpeg1();
#endif /* GL_SGIX_mpeg1 */
#ifdef GL_SGIX_mpeg2
  _glewInfo_GL_SGIX_mpeg2();
#endif /* GL_SGIX_mpeg2 */
#ifdef GL_SGIX_nonlinear_lighting_pervertex
  _glewInfo_GL_SGIX_nonlinear_lighting_pervertex();
#endif /* GL_SGIX_nonlinear_lighting_pervertex */
#ifdef GL_SGIX_nurbs_eval
  _glewInfo_GL_SGIX_nurbs_eval();
#endif /* GL_SGIX_nurbs_eval */
#ifdef GL_SGIX_occlusion_instrument
  _glewInfo_GL_SGIX_occlusion_instrument();
#endif /* GL_SGIX_occlusion_instrument */
#ifdef GL_SGIX_packed_6bytes
  _glewInfo_GL_SGIX_packed_6bytes();
#endif /* GL_SGIX_packed_6bytes */
#ifdef GL_SGIX_pixel_texture
  _glewInfo_GL_SGIX_pixel_texture();
#endif /* GL_SGIX_pixel_texture */
#ifdef GL_SGIX_pixel_texture_bits
  _glewInfo_GL_SGIX_pixel_texture_bits();
#endif /* GL_SGIX_pixel_texture_bits */
#ifdef GL_SGIX_pixel_texture_lod
  _glewInfo_GL_SGIX_pixel_texture_lod();
#endif /* GL_SGIX_pixel_texture_lod */
#ifdef GL_SGIX_pixel_tiles
  _glewInfo_GL_SGIX_pixel_tiles();
#endif /* GL_SGIX_pixel_tiles */
#ifdef GL_SGIX_polynomial_ffd
  _glewInfo_GL_SGIX_polynomial_ffd();
#endif /* GL_SGIX_polynomial_ffd */
#ifdef GL_SGIX_quad_mesh
  _glewInfo_GL_SGIX_quad_mesh();
#endif /* GL_SGIX_quad_mesh */
#ifdef GL_SGIX_reference_plane
  _glewInfo_GL_SGIX_reference_plane();
#endif /* GL_SGIX_reference_plane */
#ifdef GL_SGIX_resample
  _glewInfo_GL_SGIX_resample();
#endif /* GL_SGIX_resample */
#ifdef GL_SGIX_scalebias_hint
  _glewInfo_GL_SGIX_scalebias_hint();
#endif /* GL_SGIX_scalebias_hint */
#ifdef GL_SGIX_shadow
  _glewInfo_GL_SGIX_shadow();
#endif /* GL_SGIX_shadow */
#ifdef GL_SGIX_shadow_ambient
  _glewInfo_GL_SGIX_shadow_ambient();
#endif /* GL_SGIX_shadow_ambient */
#ifdef GL_SGIX_slim
  _glewInfo_GL_SGIX_slim();
#endif /* GL_SGIX_slim */
#ifdef GL_SGIX_spotlight_cutoff
  _glewInfo_GL_SGIX_spotlight_cutoff();
#endif /* GL_SGIX_spotlight_cutoff */
#ifdef GL_SGIX_sprite
  _glewInfo_GL_SGIX_sprite();
#endif /* GL_SGIX_sprite */
#ifdef GL_SGIX_subdiv_patch
  _glewInfo_GL_SGIX_subdiv_patch();
#endif /* GL_SGIX_subdiv_patch */
#ifdef GL_SGIX_subsample
  _glewInfo_GL_SGIX_subsample();
#endif /* GL_SGIX_subsample */
#ifdef GL_SGIX_tag_sample_buffer
  _glewInfo_GL_SGIX_tag_sample_buffer();
#endif /* GL_SGIX_tag_sample_buffer */
#ifdef GL_SGIX_texture_add_env
  _glewInfo_GL_SGIX_texture_add_env();
#endif /* GL_SGIX_texture_add_env */
#ifdef GL_SGIX_texture_coordinate_clamp
  _glewInfo_GL_SGIX_texture_coordinate_clamp();
#endif /* GL_SGIX_texture_coordinate_clamp */
#ifdef GL_SGIX_texture_lod_bias
  _glewInfo_GL_SGIX_texture_lod_bias();
#endif /* GL_SGIX_texture_lod_bias */
#ifdef GL_SGIX_texture_mipmap_anisotropic
  _glewInfo_GL_SGIX_texture_mipmap_anisotropic();
#endif /* GL_SGIX_texture_mipmap_anisotropic */
#ifdef GL_SGIX_texture_multi_buffer
  _glewInfo_GL_SGIX_texture_multi_buffer();
#endif /* GL_SGIX_texture_multi_buffer */
#ifdef GL_SGIX_texture_phase
  _glewInfo_GL_SGIX_texture_phase();
#endif /* GL_SGIX_texture_phase */
#ifdef GL_SGIX_texture_range
  _glewInfo_GL_SGIX_texture_range();
#endif /* GL_SGIX_texture_range */
#ifdef GL_SGIX_texture_scale_bias
  _glewInfo_GL_SGIX_texture_scale_bias();
#endif /* GL_SGIX_texture_scale_bias */
#ifdef GL_SGIX_texture_supersample
  _glewInfo_GL_SGIX_texture_supersample();
#endif /* GL_SGIX_texture_supersample */
#ifdef GL_SGIX_vector_ops
  _glewInfo_GL_SGIX_vector_ops();
#endif /* GL_SGIX_vector_ops */
#ifdef GL_SGIX_vertex_array_object
  _glewInfo_GL_SGIX_vertex_array_object();
#endif /* GL_SGIX_vertex_array_object */
#ifdef GL_SGIX_vertex_preclip
  _glewInfo_GL_SGIX_vertex_preclip();
#endif /* GL_SGIX_vertex_preclip */
#ifdef GL_SGIX_vertex_preclip_hint
  _glewInfo_GL_SGIX_vertex_preclip_hint();
#endif /* GL_SGIX_vertex_preclip_hint */
#ifdef GL_SGIX_ycrcb
  _glewInfo_GL_SGIX_ycrcb();
#endif /* GL_SGIX_ycrcb */
#ifdef GL_SGIX_ycrcb_subsample
  _glewInfo_GL_SGIX_ycrcb_subsample();
#endif /* GL_SGIX_ycrcb_subsample */
#ifdef GL_SGIX_ycrcba
  _glewInfo_GL_SGIX_ycrcba();
#endif /* GL_SGIX_ycrcba */
#ifdef GL_SGI_color_matrix
  _glewInfo_GL_SGI_color_matrix();
#endif /* GL_SGI_color_matrix */
#ifdef GL_SGI_color_table
  _glewInfo_GL_SGI_color_table();
#endif /* GL_SGI_color_table */
#ifdef GL_SGI_complex
  _glewInfo_GL_SGI_complex();
#endif /* GL_SGI_complex */
#ifdef GL_SGI_complex_type
  _glewInfo_GL_SGI_complex_type();
#endif /* GL_SGI_complex_type */
#ifdef GL_SGI_fft
  _glewInfo_GL_SGI_fft();
#endif /* GL_SGI_fft */
#ifdef GL_SGI_texture_color_table
  _glewInfo_GL_SGI_texture_color_table();
#endif /* GL_SGI_texture_color_table */
#ifdef GL_SUNX_constant_data
  _glewInfo_GL_SUNX_constant_data();
#endif /* GL_SUNX_constant_data */
#ifdef GL_SUN_convolution_border_modes
  _glewInfo_GL_SUN_convolution_border_modes();
#endif /* GL_SUN_convolution_border_modes */
#ifdef GL_SUN_global_alpha
  _glewInfo_GL_SUN_global_alpha();
#endif /* GL_SUN_global_alpha */
#ifdef GL_SUN_mesh_array
  _glewInfo_GL_SUN_mesh_array();
#endif /* GL_SUN_mesh_array */
#ifdef GL_SUN_read_video_pixels
  _glewInfo_GL_SUN_read_video_pixels();
#endif /* GL_SUN_read_video_pixels */
#ifdef GL_SUN_slice_accum
  _glewInfo_GL_SUN_slice_accum();
#endif /* GL_SUN_slice_accum */
#ifdef GL_SUN_triangle_list
  _glewInfo_GL_SUN_triangle_list();
#endif /* GL_SUN_triangle_list */
#ifdef GL_SUN_vertex
  _glewInfo_GL_SUN_vertex();
#endif /* GL_SUN_vertex */
#ifdef GL_VIV_shader_binary
  _glewInfo_GL_VIV_shader_binary();
#endif /* GL_VIV_shader_binary */
#ifdef GL_WIN_phong_shading
  _glewInfo_GL_WIN_phong_shading();
#endif /* GL_WIN_phong_shading */
#ifdef GL_WIN_scene_markerXXX
  _glewInfo_GL_WIN_scene_markerXXX();
#endif /* GL_WIN_scene_markerXXX */
#ifdef GL_WIN_specular_fog
  _glewInfo_GL_WIN_specular_fog();
#endif /* GL_WIN_specular_fog */
#ifdef GL_WIN_swap_hint
  _glewInfo_GL_WIN_swap_hint();
#endif /* GL_WIN_swap_hint */
}

/* ------------------------------------------------------------------------ */

#if defined(_WIN32) && !defined(GLEW_EGL) && !defined(GLEW_OSMESA)

static void wglewInfo ()
{
#ifdef WGL_3DFX_multisample
  _glewInfo_WGL_3DFX_multisample();
#endif /* WGL_3DFX_multisample */
#ifdef WGL_3DL_stereo_control
  _glewInfo_WGL_3DL_stereo_control();
#endif /* WGL_3DL_stereo_control */
#ifdef WGL_AMD_gpu_association
  _glewInfo_WGL_AMD_gpu_association();
#endif /* WGL_AMD_gpu_association */
#ifdef WGL_ARB_buffer_region
  _glewInfo_WGL_ARB_buffer_region();
#endif /* WGL_ARB_buffer_region */
#ifdef WGL_ARB_context_flush_control
  _glewInfo_WGL_ARB_context_flush_control();
#endif /* WGL_ARB_context_flush_control */
#ifdef WGL_ARB_create_context
  _glewInfo_WGL_ARB_create_context();
#endif /* WGL_ARB_create_context */
#ifdef WGL_ARB_create_context_no_error
  _glewInfo_WGL_ARB_create_context_no_error();
#endif /* WGL_ARB_create_context_no_error */
#ifdef WGL_ARB_create_context_profile
  _glewInfo_WGL_ARB_create_context_profile();
#endif /* WGL_ARB_create_context_profile */
#ifdef WGL_ARB_create_context_robustness
  _glewInfo_WGL_ARB_create_context_robustness();
#endif /* WGL_ARB_create_context_robustness */
#ifdef WGL_ARB_extensions_string
  _glewInfo_WGL_ARB_extensions_string();
#endif /* WGL_ARB_extensions_string */
#ifdef WGL_ARB_framebuffer_sRGB
  _glewInfo_WGL_ARB_framebuffer_sRGB();
#endif /* WGL_ARB_framebuffer_sRGB */
#ifdef WGL_ARB_make_current_read
  _glewInfo_WGL_ARB_make_current_read();
#endif /* WGL_ARB_make_current_read */
#ifdef WGL_ARB_multisample
  _glewInfo_WGL_ARB_multisample();
#endif /* WGL_ARB_multisample */
#ifdef WGL_ARB_pbuffer
  _glewInfo_WGL_ARB_pbuffer();
#endif /* WGL_ARB_pbuffer */
#ifdef WGL_ARB_pixel_format
  _glewInfo_WGL_ARB_pixel_format();
#endif /* WGL_ARB_pixel_format */
#ifdef WGL_ARB_pixel_format_float
  _glewInfo_WGL_ARB_pixel_format_float();
#endif /* WGL_ARB_pixel_format_float */
#ifdef WGL_ARB_render_texture
  _glewInfo_WGL_ARB_render_texture();
#endif /* WGL_ARB_render_texture */
#ifdef WGL_ARB_robustness_application_isolation
  _glewInfo_WGL_ARB_robustness_application_isolation();
#endif /* WGL_ARB_robustness_application_isolation */
#ifdef WGL_ARB_robustness_share_group_isolation
  _glewInfo_WGL_ARB_robustness_share_group_isolation();
#endif /* WGL_ARB_robustness_share_group_isolation */
#ifdef WGL_ATI_pixel_format_float
  _glewInfo_WGL_ATI_pixel_format_float();
#endif /* WGL_ATI_pixel_format_float */
#ifdef WGL_ATI_render_texture_rectangle
  _glewInfo_WGL_ATI_render_texture_rectangle();
#endif /* WGL_ATI_render_texture_rectangle */
#ifdef WGL_EXT_colorspace
  _glewInfo_WGL_EXT_colorspace();
#endif /* WGL_EXT_colorspace */
#ifdef WGL_EXT_create_context_es2_profile
  _glewInfo_WGL_EXT_create_context_es2_profile();
#endif /* WGL_EXT_create_context_es2_profile */
#ifdef WGL_EXT_create_context_es_profile
  _glewInfo_WGL_EXT_create_context_es_profile();
#endif /* WGL_EXT_create_context_es_profile */
#ifdef WGL_EXT_depth_float
  _glewInfo_WGL_EXT_depth_float();
#endif /* WGL_EXT_depth_float */
#ifdef WGL_EXT_display_color_table
  _glewInfo_WGL_EXT_display_color_table();
#endif /* WGL_EXT_display_color_table */
#ifdef WGL_EXT_extensions_string
  _glewInfo_WGL_EXT_extensions_string();
#endif /* WGL_EXT_extensions_string */
#ifdef WGL_EXT_framebuffer_sRGB
  _glewInfo_WGL_EXT_framebuffer_sRGB();
#endif /* WGL_EXT_framebuffer_sRGB */
#ifdef WGL_EXT_make_current_read
  _glewInfo_WGL_EXT_make_current_read();
#endif /* WGL_EXT_make_current_read */
#ifdef WGL_EXT_multisample
  _glewInfo_WGL_EXT_multisample();
#endif /* WGL_EXT_multisample */
#ifdef WGL_EXT_pbuffer
  _glewInfo_WGL_EXT_pbuffer();
#endif /* WGL_EXT_pbuffer */
#ifdef WGL_EXT_pixel_format
  _glewInfo_WGL_EXT_pixel_format();
#endif /* WGL_EXT_pixel_format */
#ifdef WGL_EXT_pixel_format_packed_float
  _glewInfo_WGL_EXT_pixel_format_packed_float();
#endif /* WGL_EXT_pixel_format_packed_float */
#ifdef WGL_EXT_swap_control
  _glewInfo_WGL_EXT_swap_control();
#endif /* WGL_EXT_swap_control */
#ifdef WGL_EXT_swap_control_tear
  _glewInfo_WGL_EXT_swap_control_tear();
#endif /* WGL_EXT_swap_control_tear */
#ifdef WGL_I3D_digital_video_control
  _glewInfo_WGL_I3D_digital_video_control();
#endif /* WGL_I3D_digital_video_control */
#ifdef WGL_I3D_gamma
  _glewInfo_WGL_I3D_gamma();
#endif /* WGL_I3D_gamma */
#ifdef WGL_I3D_genlock
  _glewInfo_WGL_I3D_genlock();
#endif /* WGL_I3D_genlock */
#ifdef WGL_I3D_image_buffer
  _glewInfo_WGL_I3D_image_buffer();
#endif /* WGL_I3D_image_buffer */
#ifdef WGL_I3D_swap_frame_lock
  _glewInfo_WGL_I3D_swap_frame_lock();
#endif /* WGL_I3D_swap_frame_lock */
#ifdef WGL_I3D_swap_frame_usage
  _glewInfo_WGL_I3D_swap_frame_usage();
#endif /* WGL_I3D_swap_frame_usage */
#ifdef WGL_NV_DX_interop
  _glewInfo_WGL_NV_DX_interop();
#endif /* WGL_NV_DX_interop */
#ifdef WGL_NV_DX_interop2
  _glewInfo_WGL_NV_DX_interop2();
#endif /* WGL_NV_DX_interop2 */
#ifdef WGL_NV_copy_image
  _glewInfo_WGL_NV_copy_image();
#endif /* WGL_NV_copy_image */
#ifdef WGL_NV_delay_before_swap
  _glewInfo_WGL_NV_delay_before_swap();
#endif /* WGL_NV_delay_before_swap */
#ifdef WGL_NV_float_buffer
  _glewInfo_WGL_NV_float_buffer();
#endif /* WGL_NV_float_buffer */
#ifdef WGL_NV_gpu_affinity
  _glewInfo_WGL_NV_gpu_affinity();
#endif /* WGL_NV_gpu_affinity */
#ifdef WGL_NV_multigpu_context
  _glewInfo_WGL_NV_multigpu_context();
#endif /* WGL_NV_multigpu_context */
#ifdef WGL_NV_multisample_coverage
  _glewInfo_WGL_NV_multisample_coverage();
#endif /* WGL_NV_multisample_coverage */
#ifdef WGL_NV_present_video
  _glewInfo_WGL_NV_present_video();
#endif /* WGL_NV_present_video */
#ifdef WGL_NV_render_depth_texture
  _glewInfo_WGL_NV_render_depth_texture();
#endif /* WGL_NV_render_depth_texture */
#ifdef WGL_NV_render_texture_rectangle
  _glewInfo_WGL_NV_render_texture_rectangle();
#endif /* WGL_NV_render_texture_rectangle */
#ifdef WGL_NV_swap_group
  _glewInfo_WGL_NV_swap_group();
#endif /* WGL_NV_swap_group */
#ifdef WGL_NV_vertex_array_range
  _glewInfo_WGL_NV_vertex_array_range();
#endif /* WGL_NV_vertex_array_range */
#ifdef WGL_NV_video_capture
  _glewInfo_WGL_NV_video_capture();
#endif /* WGL_NV_video_capture */
#ifdef WGL_NV_video_output
  _glewInfo_WGL_NV_video_output();
#endif /* WGL_NV_video_output */
#ifdef WGL_OML_sync_control
  _glewInfo_WGL_OML_sync_control();
#endif /* WGL_OML_sync_control */
}

#elif !defined(GLEW_EGL) && !defined(GLEW_OSMESA) /* _UNIX */

static void glxewInfo ()
{
#ifdef GLX_VERSION_1_2
  _glewInfo_GLX_VERSION_1_2();
#endif /* GLX_VERSION_1_2 */
#ifdef GLX_VERSION_1_3
  _glewInfo_GLX_VERSION_1_3();
#endif /* GLX_VERSION_1_3 */
#ifdef GLX_VERSION_1_4
  _glewInfo_GLX_VERSION_1_4();
#endif /* GLX_VERSION_1_4 */
#ifdef GLX_3DFX_multisample
  _glewInfo_GLX_3DFX_multisample();
#endif /* GLX_3DFX_multisample */
#ifdef GLX_AMD_gpu_association
  _glewInfo_GLX_AMD_gpu_association();
#endif /* GLX_AMD_gpu_association */
#ifdef GLX_ARB_context_flush_control
  _glewInfo_GLX_ARB_context_flush_control();
#endif /* GLX_ARB_context_flush_control */
#ifdef GLX_ARB_create_context
  _glewInfo_GLX_ARB_create_context();
#endif /* GLX_ARB_create_context */
#ifdef GLX_ARB_create_context_no_error
  _glewInfo_GLX_ARB_create_context_no_error();
#endif /* GLX_ARB_create_context_no_error */
#ifdef GLX_ARB_create_context_profile
  _glewInfo_GLX_ARB_create_context_profile();
#endif /* GLX_ARB_create_context_profile */
#ifdef GLX_ARB_create_context_robustness
  _glewInfo_GLX_ARB_create_context_robustness();
#endif /* GLX_ARB_create_context_robustness */
#ifdef GLX_ARB_fbconfig_float
  _glewInfo_GLX_ARB_fbconfig_float();
#endif /* GLX_ARB_fbconfig_float */
#ifdef GLX_ARB_framebuffer_sRGB
  _glewInfo_GLX_ARB_framebuffer_sRGB();
#endif /* GLX_ARB_framebuffer_sRGB */
#ifdef GLX_ARB_get_proc_address
  _glewInfo_GLX_ARB_get_proc_address();
#endif /* GLX_ARB_get_proc_address */
#ifdef GLX_ARB_multisample
  _glewInfo_GLX_ARB_multisample();
#endif /* GLX_ARB_multisample */
#ifdef GLX_ARB_robustness_application_isolation
  _glewInfo_GLX_ARB_robustness_application_isolation();
#endif /* GLX_ARB_robustness_application_isolation */
#ifdef GLX_ARB_robustness_share_group_isolation
  _glewInfo_GLX_ARB_robustness_share_group_isolation();
#endif /* GLX_ARB_robustness_share_group_isolation */
#ifdef GLX_ARB_vertex_buffer_object
  _glewInfo_GLX_ARB_vertex_buffer_object();
#endif /* GLX_ARB_vertex_buffer_object */
#ifdef GLX_ATI_pixel_format_float
  _glewInfo_GLX_ATI_pixel_format_float();
#endif /* GLX_ATI_pixel_format_float */
#ifdef GLX_ATI_render_texture
  _glewInfo_GLX_ATI_render_texture();
#endif /* GLX_ATI_render_texture */
#ifdef GLX_EXT_buffer_age
  _glewInfo_GLX_EXT_buffer_age();
#endif /* GLX_EXT_buffer_age */
#ifdef GLX_EXT_context_priority
  _glewInfo_GLX_EXT_context_priority();
#endif /* GLX_EXT_context_priority */
#ifdef GLX_EXT_create_context_es2_profile
  _glewInfo_GLX_EXT_create_context_es2_profile();
#endif /* GLX_EXT_create_context_es2_profile */
#ifdef GLX_EXT_create_context_es_profile
  _glewInfo_GLX_EXT_create_context_es_profile();
#endif /* GLX_EXT_create_context_es_profile */
#ifdef GLX_EXT_fbconfig_packed_float
  _glewInfo_GLX_EXT_fbconfig_packed_float();
#endif /* GLX_EXT_fbconfig_packed_float */
#ifdef GLX_EXT_framebuffer_sRGB
  _glewInfo_GLX_EXT_framebuffer_sRGB();
#endif /* GLX_EXT_framebuffer_sRGB */
#ifdef GLX_EXT_import_context
  _glewInfo_GLX_EXT_import_context();
#endif /* GLX_EXT_import_context */
#ifdef GLX_EXT_libglvnd
  _glewInfo_GLX_EXT_libglvnd();
#endif /* GLX_EXT_libglvnd */
#ifdef GLX_EXT_no_config_context
  _glewInfo_GLX_EXT_no_config_context();
#endif /* GLX_EXT_no_config_context */
#ifdef GLX_EXT_scene_marker
  _glewInfo_GLX_EXT_scene_marker();
#endif /* GLX_EXT_scene_marker */
#ifdef GLX_EXT_stereo_tree
  _glewInfo_GLX_EXT_stereo_tree();
#endif /* GLX_EXT_stereo_tree */
#ifdef GLX_EXT_swap_control
  _glewInfo_GLX_EXT_swap_control();
#endif /* GLX_EXT_swap_control */
#ifdef GLX_EXT_swap_control_tear
  _glewInfo_GLX_EXT_swap_control_tear();
#endif /* GLX_EXT_swap_control_tear */
#ifdef GLX_EXT_texture_from_pixmap
  _glewInfo_GLX_EXT_texture_from_pixmap();
#endif /* GLX_EXT_texture_from_pixmap */
#ifdef GLX_EXT_visual_info
  _glewInfo_GLX_EXT_visual_info();
#endif /* GLX_EXT_visual_info */
#ifdef GLX_EXT_visual_rating
  _glewInfo_GLX_EXT_visual_rating();
#endif /* GLX_EXT_visual_rating */
#ifdef GLX_INTEL_swap_event
  _glewInfo_GLX_INTEL_swap_event();
#endif /* GLX_INTEL_swap_event */
#ifdef GLX_MESA_agp_offset
  _glewInfo_GLX_MESA_agp_offset();
#endif /* GLX_MESA_agp_offset */
#ifdef GLX_MESA_copy_sub_buffer
  _glewInfo_GLX_MESA_copy_sub_buffer();
#endif /* GLX_MESA_copy_sub_buffer */
#ifdef GLX_MESA_pixmap_colormap
  _glewInfo_GLX_MESA_pixmap_colormap();
#endif /* GLX_MESA_pixmap_colormap */
#ifdef GLX_MESA_query_renderer
  _glewInfo_GLX_MESA_query_renderer();
#endif /* GLX_MESA_query_renderer */
#ifdef GLX_MESA_release_buffers
  _glewInfo_GLX_MESA_release_buffers();
#endif /* GLX_MESA_release_buffers */
#ifdef GLX_MESA_set_3dfx_mode
  _glewInfo_GLX_MESA_set_3dfx_mode();
#endif /* GLX_MESA_set_3dfx_mode */
#ifdef GLX_MESA_swap_control
  _glewInfo_GLX_MESA_swap_control();
#endif /* GLX_MESA_swap_control */
#ifdef GLX_NV_copy_buffer
  _glewInfo_GLX_NV_copy_buffer();
#endif /* GLX_NV_copy_buffer */
#ifdef GLX_NV_copy_image
  _glewInfo_GLX_NV_copy_image();
#endif /* GLX_NV_copy_image */
#ifdef GLX_NV_delay_before_swap
  _glewInfo_GLX_NV_delay_before_swap();
#endif /* GLX_NV_delay_before_swap */
#ifdef GLX_NV_float_buffer
  _glewInfo_GLX_NV_float_buffer();
#endif /* GLX_NV_float_buffer */
#ifdef GLX_NV_multigpu_context
  _glewInfo_GLX_NV_multigpu_context();
#endif /* GLX_NV_multigpu_context */
#ifdef GLX_NV_multisample_coverage
  _glewInfo_GLX_NV_multisample_coverage();
#endif /* GLX_NV_multisample_coverage */
#ifdef GLX_NV_present_video
  _glewInfo_GLX_NV_present_video();
#endif /* GLX_NV_present_video */
#ifdef GLX_NV_robustness_video_memory_purge
  _glewInfo_GLX_NV_robustness_video_memory_purge();
#endif /* GLX_NV_robustness_video_memory_purge */
#ifdef GLX_NV_swap_group
  _glewInfo_GLX_NV_swap_group();
#endif /* GLX_NV_swap_group */
#ifdef GLX_NV_vertex_array_range
  _glewInfo_GLX_NV_vertex_array_range();
#endif /* GLX_NV_vertex_array_range */
#ifdef GLX_NV_video_capture
  _glewInfo_GLX_NV_video_capture();
#endif /* GLX_NV_video_capture */
#ifdef GLX_NV_video_out
  _glewInfo_GLX_NV_video_out();
#endif /* GLX_NV_video_out */
#ifdef GLX_OML_swap_method
  _glewInfo_GLX_OML_swap_method();
#endif /* GLX_OML_swap_method */
#ifdef GLX_OML_sync_control
  _glewInfo_GLX_OML_sync_control();
#endif /* GLX_OML_sync_control */
#ifdef GLX_SGIS_blended_overlay
  _glewInfo_GLX_SGIS_blended_overlay();
#endif /* GLX_SGIS_blended_overlay */
#ifdef GLX_SGIS_color_range
  _glewInfo_GLX_SGIS_color_range();
#endif /* GLX_SGIS_color_range */
#ifdef GLX_SGIS_multisample
  _glewInfo_GLX_SGIS_multisample();
#endif /* GLX_SGIS_multisample */
#ifdef GLX_SGIS_shared_multisample
  _glewInfo_GLX_SGIS_shared_multisample();
#endif /* GLX_SGIS_shared_multisample */
#ifdef GLX_SGIX_fbconfig
  _glewInfo_GLX_SGIX_fbconfig();
#endif /* GLX_SGIX_fbconfig */
#ifdef GLX_SGIX_hyperpipe
  _glewInfo_GLX_SGIX_hyperpipe();
#endif /* GLX_SGIX_hyperpipe */
#ifdef GLX_SGIX_pbuffer
  _glewInfo_GLX_SGIX_pbuffer();
#endif /* GLX_SGIX_pbuffer */
#ifdef GLX_SGIX_swap_barrier
  _glewInfo_GLX_SGIX_swap_barrier();
#endif /* GLX_SGIX_swap_barrier */
#ifdef GLX_SGIX_swap_group
  _glewInfo_GLX_SGIX_swap_group();
#endif /* GLX_SGIX_swap_group */
#ifdef GLX_SGIX_video_resize
  _glewInfo_GLX_SGIX_video_resize();
#endif /* GLX_SGIX_video_resize */
#ifdef GLX_SGIX_visual_select_group
  _glewInfo_GLX_SGIX_visual_select_group();
#endif /* GLX_SGIX_visual_select_group */
#ifdef GLX_SGI_cushion
  _glewInfo_GLX_SGI_cushion();
#endif /* GLX_SGI_cushion */
#ifdef GLX_SGI_make_current_read
  _glewInfo_GLX_SGI_make_current_read();
#endif /* GLX_SGI_make_current_read */
#ifdef GLX_SGI_swap_control
  _glewInfo_GLX_SGI_swap_control();
#endif /* GLX_SGI_swap_control */
#ifdef GLX_SGI_video_sync
  _glewInfo_GLX_SGI_video_sync();
#endif /* GLX_SGI_video_sync */
#ifdef GLX_SUN_get_transparent_index
  _glewInfo_GLX_SUN_get_transparent_index();
#endif /* GLX_SUN_get_transparent_index */
#ifdef GLX_SUN_video_resize
  _glewInfo_GLX_SUN_video_resize();
#endif /* GLX_SUN_video_resize */
}

#elif defined(GLEW_EGL)

static void eglewInfo ()
{
#ifdef EGL_VERSION_1_0
  _glewInfo_EGL_VERSION_1_0();
#endif /* EGL_VERSION_1_0 */
#ifdef EGL_VERSION_1_1
  _glewInfo_EGL_VERSION_1_1();
#endif /* EGL_VERSION_1_1 */
#ifdef EGL_VERSION_1_2
  _glewInfo_EGL_VERSION_1_2();
#endif /* EGL_VERSION_1_2 */
#ifdef EGL_VERSION_1_3
  _glewInfo_EGL_VERSION_1_3();
#endif /* EGL_VERSION_1_3 */
#ifdef EGL_VERSION_1_4
  _glewInfo_EGL_VERSION_1_4();
#endif /* EGL_VERSION_1_4 */
#ifdef EGL_VERSION_1_5
  _glewInfo_EGL_VERSION_1_5();
#endif /* EGL_VERSION_1_5 */
#ifdef EGL_ANDROID_GLES_layers
  _glewInfo_EGL_ANDROID_GLES_layers();
#endif /* EGL_ANDROID_GLES_layers */
#ifdef EGL_ANDROID_blob_cache
  _glewInfo_EGL_ANDROID_blob_cache();
#endif /* EGL_ANDROID_blob_cache */
#ifdef EGL_ANDROID_create_native_client_buffer
  _glewInfo_EGL_ANDROID_create_native_client_buffer();
#endif /* EGL_ANDROID_create_native_client_buffer */
#ifdef EGL_ANDROID_framebuffer_target
  _glewInfo_EGL_ANDROID_framebuffer_target();
#endif /* EGL_ANDROID_framebuffer_target */
#ifdef EGL_ANDROID_front_buffer_auto_refresh
  _glewInfo_EGL_ANDROID_front_buffer_auto_refresh();
#endif /* EGL_ANDROID_front_buffer_auto_refresh */
#ifdef EGL_ANDROID_get_frame_timestamps
  _glewInfo_EGL_ANDROID_get_frame_timestamps();
#endif /* EGL_ANDROID_get_frame_timestamps */
#ifdef EGL_ANDROID_get_native_client_buffer
  _glewInfo_EGL_ANDROID_get_native_client_buffer();
#endif /* EGL_ANDROID_get_native_client_buffer */
#ifdef EGL_ANDROID_image_native_buffer
  _glewInfo_EGL_ANDROID_image_native_buffer();
#endif /* EGL_ANDROID_image_native_buffer */
#ifdef EGL_ANDROID_native_fence_sync
  _glewInfo_EGL_ANDROID_native_fence_sync();
#endif /* EGL_ANDROID_native_fence_sync */
#ifdef EGL_ANDROID_presentation_time
  _glewInfo_EGL_ANDROID_presentation_time();
#endif /* EGL_ANDROID_presentation_time */
#ifdef EGL_ANDROID_recordable
  _glewInfo_EGL_ANDROID_recordable();
#endif /* EGL_ANDROID_recordable */
#ifdef EGL_ANGLE_d3d_share_handle_client_buffer
  _glewInfo_EGL_ANGLE_d3d_share_handle_client_buffer();
#endif /* EGL_ANGLE_d3d_share_handle_client_buffer */
#ifdef EGL_ANGLE_device_d3d
  _glewInfo_EGL_ANGLE_device_d3d();
#endif /* EGL_ANGLE_device_d3d */
#ifdef EGL_ANGLE_query_surface_pointer
  _glewInfo_EGL_ANGLE_query_surface_pointer();
#endif /* EGL_ANGLE_query_surface_pointer */
#ifdef EGL_ANGLE_surface_d3d_texture_2d_share_handle
  _glewInfo_EGL_ANGLE_surface_d3d_texture_2d_share_handle();
#endif /* EGL_ANGLE_surface_d3d_texture_2d_share_handle */
#ifdef EGL_ANGLE_window_fixed_size
  _glewInfo_EGL_ANGLE_window_fixed_size();
#endif /* EGL_ANGLE_window_fixed_size */
#ifdef EGL_ARM_image_format
  _glewInfo_EGL_ARM_image_format();
#endif /* EGL_ARM_image_format */
#ifdef EGL_ARM_implicit_external_sync
  _glewInfo_EGL_ARM_implicit_external_sync();
#endif /* EGL_ARM_implicit_external_sync */
#ifdef EGL_ARM_pixmap_multisample_discard
  _glewInfo_EGL_ARM_pixmap_multisample_discard();
#endif /* EGL_ARM_pixmap_multisample_discard */
#ifdef EGL_EXT_bind_to_front
  _glewInfo_EGL_EXT_bind_to_front();
#endif /* EGL_EXT_bind_to_front */
#ifdef EGL_EXT_buffer_age
  _glewInfo_EGL_EXT_buffer_age();
#endif /* EGL_EXT_buffer_age */
#ifdef EGL_EXT_client_extensions
  _glewInfo_EGL_EXT_client_extensions();
#endif /* EGL_EXT_client_extensions */
#ifdef EGL_EXT_client_sync
  _glewInfo_EGL_EXT_client_sync();
#endif /* EGL_EXT_client_sync */
#ifdef EGL_EXT_compositor
  _glewInfo_EGL_EXT_compositor();
#endif /* EGL_EXT_compositor */
#ifdef EGL_EXT_create_context_robustness
  _glewInfo_EGL_EXT_create_context_robustness();
#endif /* EGL_EXT_create_context_robustness */
#ifdef EGL_EXT_device_base
  _glewInfo_EGL_EXT_device_base();
#endif /* EGL_EXT_device_base */
#ifdef EGL_EXT_device_drm
  _glewInfo_EGL_EXT_device_drm();
#endif /* EGL_EXT_device_drm */
#ifdef EGL_EXT_device_enumeration
  _glewInfo_EGL_EXT_device_enumeration();
#endif /* EGL_EXT_device_enumeration */
#ifdef EGL_EXT_device_openwf
  _glewInfo_EGL_EXT_device_openwf();
#endif /* EGL_EXT_device_openwf */
#ifdef EGL_EXT_device_query
  _glewInfo_EGL_EXT_device_query();
#endif /* EGL_EXT_device_query */
#ifdef EGL_EXT_gl_colorspace_bt2020_linear
  _glewInfo_EGL_EXT_gl_colorspace_bt2020_linear();
#endif /* EGL_EXT_gl_colorspace_bt2020_linear */
#ifdef EGL_EXT_gl_colorspace_bt2020_pq
  _glewInfo_EGL_EXT_gl_colorspace_bt2020_pq();
#endif /* EGL_EXT_gl_colorspace_bt2020_pq */
#ifdef EGL_EXT_gl_colorspace_display_p3
  _glewInfo_EGL_EXT_gl_colorspace_display_p3();
#endif /* EGL_EXT_gl_colorspace_display_p3 */
#ifdef EGL_EXT_gl_colorspace_display_p3_linear
  _glewInfo_EGL_EXT_gl_colorspace_display_p3_linear();
#endif /* EGL_EXT_gl_colorspace_display_p3_linear */
#ifdef EGL_EXT_gl_colorspace_display_p3_passthrough
  _glewInfo_EGL_EXT_gl_colorspace_display_p3_passthrough();
#endif /* EGL_EXT_gl_colorspace_display_p3_passthrough */
#ifdef EGL_EXT_gl_colorspace_scrgb
  _glewInfo_EGL_EXT_gl_colorspace_scrgb();
#endif /* EGL_EXT_gl_colorspace_scrgb */
#ifdef EGL_EXT_gl_colorspace_scrgb_linear
  _glewInfo_EGL_EXT_gl_colorspace_scrgb_linear();
#endif /* EGL_EXT_gl_colorspace_scrgb_linear */
#ifdef EGL_EXT_image_dma_buf_import
  _glewInfo_EGL_EXT_image_dma_buf_import();
#endif /* EGL_EXT_image_dma_buf_import */
#ifdef EGL_EXT_image_dma_buf_import_modifiers
  _glewInfo_EGL_EXT_image_dma_buf_import_modifiers();
#endif /* EGL_EXT_image_dma_buf_import_modifiers */
#ifdef EGL_EXT_image_gl_colorspace
  _glewInfo_EGL_EXT_image_gl_colorspace();
#endif /* EGL_EXT_image_gl_colorspace */
#ifdef EGL_EXT_image_implicit_sync_control
  _glewInfo_EGL_EXT_image_implicit_sync_control();
#endif /* EGL_EXT_image_implicit_sync_control */
#ifdef EGL_EXT_multiview_window
  _glewInfo_EGL_EXT_multiview_window();
#endif /* EGL_EXT_multiview_window */
#ifdef EGL_EXT_output_base
  _glewInfo_EGL_EXT_output_base();
#endif /* EGL_EXT_output_base */
#ifdef EGL_EXT_output_drm
  _glewInfo_EGL_EXT_output_drm();
#endif /* EGL_EXT_output_drm */
#ifdef EGL_EXT_output_openwf
  _glewInfo_EGL_EXT_output_openwf();
#endif /* EGL_EXT_output_openwf */
#ifdef EGL_EXT_pixel_format_float
  _glewInfo_EGL_EXT_pixel_format_float();
#endif /* EGL_EXT_pixel_format_float */
#ifdef EGL_EXT_platform_base
  _glewInfo_EGL_EXT_platform_base();
#endif /* EGL_EXT_platform_base */
#ifdef EGL_EXT_platform_device
  _glewInfo_EGL_EXT_platform_device();
#endif /* EGL_EXT_platform_device */
#ifdef EGL_EXT_platform_wayland
  _glewInfo_EGL_EXT_platform_wayland();
#endif /* EGL_EXT_platform_wayland */
#ifdef EGL_EXT_platform_x11
  _glewInfo_EGL_EXT_platform_x11();
#endif /* EGL_EXT_platform_x11 */
#ifdef EGL_EXT_protected_content
  _glewInfo_EGL_EXT_protected_content();
#endif /* EGL_EXT_protected_content */
#ifdef EGL_EXT_protected_surface
  _glewInfo_EGL_EXT_protected_surface();
#endif /* EGL_EXT_protected_surface */
#ifdef EGL_EXT_stream_consumer_egloutput
  _glewInfo_EGL_EXT_stream_consumer_egloutput();
#endif /* EGL_EXT_stream_consumer_egloutput */
#ifdef EGL_EXT_surface_CTA861_3_metadata
  _glewInfo_EGL_EXT_surface_CTA861_3_metadata();
#endif /* EGL_EXT_surface_CTA861_3_metadata */
#ifdef EGL_EXT_surface_SMPTE2086_metadata
  _glewInfo_EGL_EXT_surface_SMPTE2086_metadata();
#endif /* EGL_EXT_surface_SMPTE2086_metadata */
#ifdef EGL_EXT_swap_buffers_with_damage
  _glewInfo_EGL_EXT_swap_buffers_with_damage();
#endif /* EGL_EXT_swap_buffers_with_damage */
#ifdef EGL_EXT_sync_reuse
  _glewInfo_EGL_EXT_sync_reuse();
#endif /* EGL_EXT_sync_reuse */
#ifdef EGL_EXT_yuv_surface
  _glewInfo_EGL_EXT_yuv_surface();
#endif /* EGL_EXT_yuv_surface */
#ifdef EGL_HI_clientpixmap
  _glewInfo_EGL_HI_clientpixmap();
#endif /* EGL_HI_clientpixmap */
#ifdef EGL_HI_colorformats
  _glewInfo_EGL_HI_colorformats();
#endif /* EGL_HI_colorformats */
#ifdef EGL_IMG_context_priority
  _glewInfo_EGL_IMG_context_priority();
#endif /* EGL_IMG_context_priority */
#ifdef EGL_IMG_image_plane_attribs
  _glewInfo_EGL_IMG_image_plane_attribs();
#endif /* EGL_IMG_image_plane_attribs */
#ifdef EGL_KHR_cl_event
  _glewInfo_EGL_KHR_cl_event();
#endif /* EGL_KHR_cl_event */
#ifdef EGL_KHR_cl_event2
  _glewInfo_EGL_KHR_cl_event2();
#endif /* EGL_KHR_cl_event2 */
#ifdef EGL_KHR_client_get_all_proc_addresses
  _glewInfo_EGL_KHR_client_get_all_proc_addresses();
#endif /* EGL_KHR_client_get_all_proc_addresses */
#ifdef EGL_KHR_config_attribs
  _glewInfo_EGL_KHR_config_attribs();
#endif /* EGL_KHR_config_attribs */
#ifdef EGL_KHR_context_flush_control
  _glewInfo_EGL_KHR_context_flush_control();
#endif /* EGL_KHR_context_flush_control */
#ifdef EGL_KHR_create_context
  _glewInfo_EGL_KHR_create_context();
#endif /* EGL_KHR_create_context */
#ifdef EGL_KHR_create_context_no_error
  _glewInfo_EGL_KHR_create_context_no_error();
#endif /* EGL_KHR_create_context_no_error */
#ifdef EGL_KHR_debug
  _glewInfo_EGL_KHR_debug();
#endif /* EGL_KHR_debug */
#ifdef EGL_KHR_display_reference
  _glewInfo_EGL_KHR_display_reference();
#endif /* EGL_KHR_display_reference */
#ifdef EGL_KHR_fence_sync
  _glewInfo_EGL_KHR_fence_sync();
#endif /* EGL_KHR_fence_sync */
#ifdef EGL_KHR_get_all_proc_addresses
  _glewInfo_EGL_KHR_get_all_proc_addresses();
#endif /* EGL_KHR_get_all_proc_addresses */
#ifdef EGL_KHR_gl_colorspace
  _glewInfo_EGL_KHR_gl_colorspace();
#endif /* EGL_KHR_gl_colorspace */
#ifdef EGL_KHR_gl_renderbuffer_image
  _glewInfo_EGL_KHR_gl_renderbuffer_image();
#endif /* EGL_KHR_gl_renderbuffer_image */
#ifdef EGL_KHR_gl_texture_2D_image
  _glewInfo_EGL_KHR_gl_texture_2D_image();
#endif /* EGL_KHR_gl_texture_2D_image */
#ifdef EGL_KHR_gl_texture_3D_image
  _glewInfo_EGL_KHR_gl_texture_3D_image();
#endif /* EGL_KHR_gl_texture_3D_image */
#ifdef EGL_KHR_gl_texture_cubemap_image
  _glewInfo_EGL_KHR_gl_texture_cubemap_image();
#endif /* EGL_KHR_gl_texture_cubemap_image */
#ifdef EGL_KHR_image
  _glewInfo_EGL_KHR_image();
#endif /* EGL_KHR_image */
#ifdef EGL_KHR_image_base
  _glewInfo_EGL_KHR_image_base();
#endif /* EGL_KHR_image_base */
#ifdef EGL_KHR_image_pixmap
  _glewInfo_EGL_KHR_image_pixmap();
#endif /* EGL_KHR_image_pixmap */
#ifdef EGL_KHR_lock_surface
  _glewInfo_EGL_KHR_lock_surface();
#endif /* EGL_KHR_lock_surface */
#ifdef EGL_KHR_lock_surface2
  _glewInfo_EGL_KHR_lock_surface2();
#endif /* EGL_KHR_lock_surface2 */
#ifdef EGL_KHR_lock_surface3
  _glewInfo_EGL_KHR_lock_surface3();
#endif /* EGL_KHR_lock_surface3 */
#ifdef EGL_KHR_mutable_render_buffer
  _glewInfo_EGL_KHR_mutable_render_buffer();
#endif /* EGL_KHR_mutable_render_buffer */
#ifdef EGL_KHR_no_config_context
  _glewInfo_EGL_KHR_no_config_context();
#endif /* EGL_KHR_no_config_context */
#ifdef EGL_KHR_partial_update
  _glewInfo_EGL_KHR_partial_update();
#endif /* EGL_KHR_partial_update */
#ifdef EGL_KHR_platform_android
  _glewInfo_EGL_KHR_platform_android();
#endif /* EGL_KHR_platform_android */
#ifdef EGL_KHR_platform_gbm
  _glewInfo_EGL_KHR_platform_gbm();
#endif /* EGL_KHR_platform_gbm */
#ifdef EGL_KHR_platform_wayland
  _glewInfo_EGL_KHR_platform_wayland();
#endif /* EGL_KHR_platform_wayland */
#ifdef EGL_KHR_platform_x11
  _glewInfo_EGL_KHR_platform_x11();
#endif /* EGL_KHR_platform_x11 */
#ifdef EGL_KHR_reusable_sync
  _glewInfo_EGL_KHR_reusable_sync();
#endif /* EGL_KHR_reusable_sync */
#ifdef EGL_KHR_stream
  _glewInfo_EGL_KHR_stream();
#endif /* EGL_KHR_stream */
#ifdef EGL_KHR_stream_attrib
  _glewInfo_EGL_KHR_stream_attrib();
#endif /* EGL_KHR_stream_attrib */
#ifdef EGL_KHR_stream_consumer_gltexture
  _glewInfo_EGL_KHR_stream_consumer_gltexture();
#endif /* EGL_KHR_stream_consumer_gltexture */
#ifdef EGL_KHR_stream_cross_process_fd
  _glewInfo_EGL_KHR_stream_cross_process_fd();
#endif /* EGL_KHR_stream_cross_process_fd */
#ifdef EGL_KHR_stream_fifo
  _glewInfo_EGL_KHR_stream_fifo();
#endif /* EGL_KHR_stream_fifo */
#ifdef EGL_KHR_stream_producer_aldatalocator
  _glewInfo_EGL_KHR_stream_producer_aldatalocator();
#endif /* EGL_KHR_stream_producer_aldatalocator */
#ifdef EGL_KHR_stream_producer_eglsurface
  _glewInfo_EGL_KHR_stream_producer_eglsurface();
#endif /* EGL_KHR_stream_producer_eglsurface */
#ifdef EGL_KHR_surfaceless_context
  _glewInfo_EGL_KHR_surfaceless_context();
#endif /* EGL_KHR_surfaceless_context */
#ifdef EGL_KHR_swap_buffers_with_damage
  _glewInfo_EGL_KHR_swap_buffers_with_damage();
#endif /* EGL_KHR_swap_buffers_with_damage */
#ifdef EGL_KHR_vg_parent_image
  _glewInfo_EGL_KHR_vg_parent_image();
#endif /* EGL_KHR_vg_parent_image */
#ifdef EGL_KHR_wait_sync
  _glewInfo_EGL_KHR_wait_sync();
#endif /* EGL_KHR_wait_sync */
#ifdef EGL_MESA_drm_image
  _glewInfo_EGL_MESA_drm_image();
#endif /* EGL_MESA_drm_image */
#ifdef EGL_MESA_image_dma_buf_export
  _glewInfo_EGL_MESA_image_dma_buf_export();
#endif /* EGL_MESA_image_dma_buf_export */
#ifdef EGL_MESA_platform_gbm
  _glewInfo_EGL_MESA_platform_gbm();
#endif /* EGL_MESA_platform_gbm */
#ifdef EGL_MESA_platform_surfaceless
  _glewInfo_EGL_MESA_platform_surfaceless();
#endif /* EGL_MESA_platform_surfaceless */
#ifdef EGL_MESA_query_driver
  _glewInfo_EGL_MESA_query_driver();
#endif /* EGL_MESA_query_driver */
#ifdef EGL_NOK_swap_region
  _glewInfo_EGL_NOK_swap_region();
#endif /* EGL_NOK_swap_region */
#ifdef EGL_NOK_swap_region2
  _glewInfo_EGL_NOK_swap_region2();
#endif /* EGL_NOK_swap_region2 */
#ifdef EGL_NOK_texture_from_pixmap
  _glewInfo_EGL_NOK_texture_from_pixmap();
#endif /* EGL_NOK_texture_from_pixmap */
#ifdef EGL_NV_3dvision_surface
  _glewInfo_EGL_NV_3dvision_surface();
#endif /* EGL_NV_3dvision_surface */
#ifdef EGL_NV_context_priority_realtime
  _glewInfo_EGL_NV_context_priority_realtime();
#endif /* EGL_NV_context_priority_realtime */
#ifdef EGL_NV_coverage_sample
  _glewInfo_EGL_NV_coverage_sample();
#endif /* EGL_NV_coverage_sample */
#ifdef EGL_NV_coverage_sample_resolve
  _glewInfo_EGL_NV_coverage_sample_resolve();
#endif /* EGL_NV_coverage_sample_resolve */
#ifdef EGL_NV_cuda_event
  _glewInfo_EGL_NV_cuda_event();
#endif /* EGL_NV_cuda_event */
#ifdef EGL_NV_depth_nonlinear
  _glewInfo_EGL_NV_depth_nonlinear();
#endif /* EGL_NV_depth_nonlinear */
#ifdef EGL_NV_device_cuda
  _glewInfo_EGL_NV_device_cuda();
#endif /* EGL_NV_device_cuda */
#ifdef EGL_NV_native_query
  _glewInfo_EGL_NV_native_query();
#endif /* EGL_NV_native_query */
#ifdef EGL_NV_post_convert_rounding
  _glewInfo_EGL_NV_post_convert_rounding();
#endif /* EGL_NV_post_convert_rounding */
#ifdef EGL_NV_post_sub_buffer
  _glewInfo_EGL_NV_post_sub_buffer();
#endif /* EGL_NV_post_sub_buffer */
#ifdef EGL_NV_quadruple_buffer
  _glewInfo_EGL_NV_quadruple_buffer();
#endif /* EGL_NV_quadruple_buffer */
#ifdef EGL_NV_robustness_video_memory_purge
  _glewInfo_EGL_NV_robustness_video_memory_purge();
#endif /* EGL_NV_robustness_video_memory_purge */
#ifdef EGL_NV_stream_consumer_gltexture_yuv
  _glewInfo_EGL_NV_stream_consumer_gltexture_yuv();
#endif /* EGL_NV_stream_consumer_gltexture_yuv */
#ifdef EGL_NV_stream_cross_display
  _glewInfo_EGL_NV_stream_cross_display();
#endif /* EGL_NV_stream_cross_display */
#ifdef EGL_NV_stream_cross_object
  _glewInfo_EGL_NV_stream_cross_object();
#endif /* EGL_NV_stream_cross_object */
#ifdef EGL_NV_stream_cross_partition
  _glewInfo_EGL_NV_stream_cross_partition();
#endif /* EGL_NV_stream_cross_partition */
#ifdef EGL_NV_stream_cross_process
  _glewInfo_EGL_NV_stream_cross_process();
#endif /* EGL_NV_stream_cross_process */
#ifdef EGL_NV_stream_cross_system
  _glewInfo_EGL_NV_stream_cross_system();
#endif /* EGL_NV_stream_cross_system */
#ifdef EGL_NV_stream_dma
  _glewInfo_EGL_NV_stream_dma();
#endif /* EGL_NV_stream_dma */
#ifdef EGL_NV_stream_fifo_next
  _glewInfo_EGL_NV_stream_fifo_next();
#endif /* EGL_NV_stream_fifo_next */
#ifdef EGL_NV_stream_fifo_synchronous
  _glewInfo_EGL_NV_stream_fifo_synchronous();
#endif /* EGL_NV_stream_fifo_synchronous */
#ifdef EGL_NV_stream_flush
  _glewInfo_EGL_NV_stream_flush();
#endif /* EGL_NV_stream_flush */
#ifdef EGL_NV_stream_frame_limits
  _glewInfo_EGL_NV_stream_frame_limits();
#endif /* EGL_NV_stream_frame_limits */
#ifdef EGL_NV_stream_metadata
  _glewInfo_EGL_NV_stream_metadata();
#endif /* EGL_NV_stream_metadata */
#ifdef EGL_NV_stream_origin
  _glewInfo_EGL_NV_stream_origin();
#endif /* EGL_NV_stream_origin */
#ifdef EGL_NV_stream_remote
  _glewInfo_EGL_NV_stream_remote();
#endif /* EGL_NV_stream_remote */
#ifdef EGL_NV_stream_reset
  _glewInfo_EGL_NV_stream_reset();
#endif /* EGL_NV_stream_reset */
#ifdef EGL_NV_stream_socket
  _glewInfo_EGL_NV_stream_socket();
#endif /* EGL_NV_stream_socket */
#ifdef EGL_NV_stream_socket_inet
  _glewInfo_EGL_NV_stream_socket_inet();
#endif /* EGL_NV_stream_socket_inet */
#ifdef EGL_NV_stream_socket_unix
  _glewInfo_EGL_NV_stream_socket_unix();
#endif /* EGL_NV_stream_socket_unix */
#ifdef EGL_NV_stream_sync
  _glewInfo_EGL_NV_stream_sync();
#endif /* EGL_NV_stream_sync */
#ifdef EGL_NV_sync
  _glewInfo_EGL_NV_sync();
#endif /* EGL_NV_sync */
#ifdef EGL_NV_system_time
  _glewInfo_EGL_NV_system_time();
#endif /* EGL_NV_system_time */
#ifdef EGL_NV_triple_buffer
  _glewInfo_EGL_NV_triple_buffer();
#endif /* EGL_NV_triple_buffer */
#ifdef EGL_TIZEN_image_native_buffer
  _glewInfo_EGL_TIZEN_image_native_buffer();
#endif /* EGL_TIZEN_image_native_buffer */
#ifdef EGL_TIZEN_image_native_surface
  _glewInfo_EGL_TIZEN_image_native_surface();
#endif /* EGL_TIZEN_image_native_surface */
#ifdef EGL_WL_bind_wayland_display
  _glewInfo_EGL_WL_bind_wayland_display();
#endif /* EGL_WL_bind_wayland_display */
#ifdef EGL_WL_create_wayland_buffer_from_image
  _glewInfo_EGL_WL_create_wayland_buffer_from_image();
#endif /* EGL_WL_create_wayland_buffer_from_image */
}

#endif /* _WIN32 */

/* ------------------------------------------------------------------------ */

int main (int argc, char** argv)
{
  GLuint err;
  struct createParams params =
  {
#if defined(GLEW_OSMESA)
#elif defined(GLEW_EGL)
#elif defined(_WIN32)
    -1,  /* pixelformat */
#elif !defined(__HAIKU__) && !defined(__APPLE__) || defined(GLEW_APPLE_GLX)
    "",  /* display */
    -1,  /* visual */
#endif
    0,   /* major */
    0,   /* minor */
    0,   /* profile mask */
    0,   /* flags */
    0    /* experimental */
  };

#if defined(GLEW_EGL)
  typedef const GLubyte* (GLAPIENTRY * PFNGLGETSTRINGPROC) (GLenum name);
  PFNGLGETSTRINGPROC getString;
#endif

  if (glewParseArgs(argc-1, argv+1, &params))
  {
    fprintf(stderr, "Usage: glewinfo "
#if defined(GLEW_OSMESA)
#elif defined(GLEW_EGL)
#elif defined(_WIN32)
      "[-pf <pixelformat>] "
#elif !defined(__HAIKU__) && !defined(__APPLE__) || defined(GLEW_APPLE_GLX)
      "[-display <display>] "
      "[-visual <visual id>] "
#endif
      "[-version <OpenGL version>] "
      "[-profile core|compatibility] "
      "[-flag debug|forward] "
      "[-experimental]"
      "\n");
    return 1;
  }

  if (GL_TRUE == glewCreateContext(&params))
  {
    fprintf(stderr, "Error: glewCreateContext failed\n");
    glewDestroyContext();
    return 1;
  }
  glewExperimental = params.experimental ? GL_TRUE : GL_FALSE;
  err = glewInit();
  if (GLEW_OK != err)
  {
    fprintf(stderr, "Error [main]: glewInit failed: %s\n", glewGetErrorString(err));
    glewDestroyContext();
    return 1;
  }

#if defined(GLEW_EGL)
  getString = (PFNGLGETSTRINGPROC) eglGetProcAddress("glGetString");
  if (!getString)
  {
    fprintf(stderr, "Error: eglGetProcAddress failed to fetch glGetString\n");
    glewDestroyContext();
    return 1;
  }
#endif

#if defined(_WIN32)
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
  if (fopen_s(&f, "glewinfo.txt", "w") != 0)
    f = stdout;
#else
  f = fopen("glewinfo.txt", "w");
#endif
  if (f == NULL) f = stdout;
#else
  f = stdout;
#endif
  fprintf(f, "---------------------------\n");
  fprintf(f, "    GLEW Extension Info\n");
  fprintf(f, "---------------------------\n\n");
  fprintf(f, "GLEW version %s\n", glewGetString(GLEW_VERSION));
#if defined(GLEW_OSMESA)
#elif defined(GLEW_EGL)
#elif defined(_WIN32)
  fprintf(f, "Reporting capabilities of pixelformat %d\n", params.pixelformat);
#elif !defined(__APPLE__) || defined(GLEW_APPLE_GLX)
  fprintf(f, "Reporting capabilities of display %s, visual 0x%x\n",
    params.display == NULL ? getenv("DISPLAY") : params.display, params.visual);
#endif
#if defined(GLEW_EGL)
  fprintf(f, "Running on a %s from %s\n",
    getString(GL_RENDERER), getString(GL_VENDOR));
  fprintf(f, "OpenGL version %s is supported\n", getString(GL_VERSION));
#else
  fprintf(f, "Running on a %s from %s\n",
    glGetString(GL_RENDERER), glGetString(GL_VENDOR));
  fprintf(f, "OpenGL version %s is supported\n", glGetString(GL_VERSION));
#endif
  glewInfo();
#if defined(GLEW_OSMESA)
#elif defined(GLEW_EGL)
  eglewInfo();
#elif defined(_WIN32)
  wglewInfo();
#else
  glxewInfo();
#endif
  if (f != stdout) fclose(f);
  glewDestroyContext();
  return 0;
}

/* ------------------------------------------------------------------------ */

GLboolean glewParseArgs (int argc, char** argv, struct createParams *params)
{
  int p = 0;
  while (p < argc)
  {
    if (!strcmp(argv[p], "-version"))
    {
      if (++p >= argc) return GL_TRUE;
#if defined(__STDC_LIB_EXT1__) || (defined(_MSC_VER) && (_MSC_VER >= 1400))
      if (sscanf_s(argv[p++], "%d.%d", &params->major, &params->minor) != 2) return GL_TRUE;
#else
      if (sscanf(argv[p++], "%d.%d", &params->major, &params->minor) != 2) return GL_TRUE;
#endif
    }
    else if (!strcmp(argv[p], "-profile"))
    {
      if (++p >= argc) return GL_TRUE;
      if      (strcmp("core",         argv[p]) == 0) params->profile |= 1;
      else if (strcmp("compatibility",argv[p]) == 0) params->profile |= 2;
      else return GL_TRUE;
      ++p;
    }
    else if (!strcmp(argv[p], "-flag"))
    {
      if (++p >= argc) return GL_TRUE;
      if      (strcmp("debug",  argv[p]) == 0) params->flags |= 1;
      else if (strcmp("forward",argv[p]) == 0) params->flags |= 2;
      else return GL_TRUE;
      ++p;
    }
#if defined(GLEW_OSMESA)
#elif defined(GLEW_EGL)
#elif defined(_WIN32)
    else if (!strcmp(argv[p], "-pf") || !strcmp(argv[p], "-pixelformat"))
    {
      if (++p >= argc) return GL_TRUE;
      params->pixelformat = strtol(argv[p++], NULL, 0);
    }
#elif !defined(__HAIKU__) && !defined(__APPLE__) || defined(GLEW_APPLE_GLX)
    else if (!strcmp(argv[p], "-display"))
    {
      if (++p >= argc) return GL_TRUE;
      params->display = argv[p++];
    }
    else if (!strcmp(argv[p], "-visual"))
    {
      if (++p >= argc) return GL_TRUE;
      params->visual = (int)strtol(argv[p++], NULL, 0);
    }
#endif
    else if (!strcmp(argv[p], "-experimental"))
    {
      params->experimental = 1;
      ++p;
    }
    else
      return GL_TRUE;
  }
  return GL_FALSE;
}

/* ------------------------------------------------------------------------ */

#if defined(GLEW_EGL)
EGLDisplay  display;
EGLContext  ctx;

/* See: http://stackoverflow.com/questions/12662227/opengl-es2-0-offscreen-context-for-fbo-rendering */

GLboolean glewCreateContext (struct createParams *params)
{
  EGLDeviceEXT devices[1];
  EGLint numDevices;
  EGLSurface  surface;
  EGLint majorVersion, minorVersion;
  EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 1,
        EGL_GREEN_SIZE, 1,
        EGL_BLUE_SIZE, 1,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
   };
  static const EGLint contextAttribs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 2,
    EGL_NONE
  };
  static const EGLint pBufferAttribs[] = {
    EGL_WIDTH,  128,
    EGL_HEIGHT, 128,
    EGL_NONE
  };
  EGLConfig config;
  EGLint numConfig;
  EGLBoolean pBuffer;

  PFNEGLQUERYDEVICESEXTPROC       queryDevices = NULL;
  PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay = NULL;
  PFNEGLGETERRORPROC              getError = NULL;
  PFNEGLGETDISPLAYPROC            getDisplay = NULL;
  PFNEGLINITIALIZEPROC            initialize = NULL;
  PFNEGLBINDAPIPROC               bindAPI    = NULL;
  PFNEGLCHOOSECONFIGPROC          chooseConfig = NULL;
  PFNEGLCREATEWINDOWSURFACEPROC   createWindowSurface = NULL;
  PFNEGLCREATECONTEXTPROC         createContext = NULL;
  PFNEGLMAKECURRENTPROC           makeCurrent = NULL;
  PFNEGLCREATEPBUFFERSURFACEPROC  createPbufferSurface = NULL;

  /* Load necessary entry points */
  queryDevices         = (PFNEGLQUERYDEVICESEXTPROC)       eglGetProcAddress("eglQueryDevicesEXT");
  getPlatformDisplay   = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
  getError             = (PFNEGLGETERRORPROC)              eglGetProcAddress("eglGetError");
  getDisplay           = (PFNEGLGETDISPLAYPROC)            eglGetProcAddress("eglGetDisplay");
  initialize           = (PFNEGLINITIALIZEPROC)            eglGetProcAddress("eglInitialize");
  bindAPI              = (PFNEGLBINDAPIPROC)               eglGetProcAddress("eglBindAPI");
  chooseConfig         = (PFNEGLCHOOSECONFIGPROC)          eglGetProcAddress("eglChooseConfig");
  createWindowSurface  = (PFNEGLCREATEWINDOWSURFACEPROC)   eglGetProcAddress("eglCreateWindowSurface");
  createPbufferSurface = (PFNEGLCREATEPBUFFERSURFACEPROC)  eglGetProcAddress("eglCreatePbufferSurface");
  createContext        = (PFNEGLCREATECONTEXTPROC)         eglGetProcAddress("eglCreateContext");
  makeCurrent          = (PFNEGLMAKECURRENTPROC)           eglGetProcAddress("eglMakeCurrent");
  if (!getError || !getDisplay || !initialize || !bindAPI || !chooseConfig || !createWindowSurface || !createContext || !makeCurrent)
    return GL_TRUE;

  pBuffer = 0;
  display = EGL_NO_DISPLAY;
  if (queryDevices && getPlatformDisplay)
  {
    queryDevices(1, devices, &numDevices);
    if (numDevices==1)
    {
      /* Nvidia EGL doesn't need X11 for p-buffer surface */
      display = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[0], 0);
      configAttribs[1] = EGL_PBUFFER_BIT;
      pBuffer = 1;
    }
  }
  if (display==EGL_NO_DISPLAY)
  {
    /* Fall-back to X11 surface, works on Mesa */
    display = getDisplay(EGL_DEFAULT_DISPLAY);
  }
  if (display == EGL_NO_DISPLAY)
    return GL_TRUE;

  eglewInit(display);

  if (bindAPI(EGL_OPENGL_API) != EGL_TRUE)
    return GL_TRUE;

  if (chooseConfig(display, configAttribs, &config, 1, &numConfig) != EGL_TRUE || (numConfig != 1))
    return GL_TRUE;

  ctx = createContext(display, config, EGL_NO_CONTEXT, pBuffer ? contextAttribs : NULL);
  if (NULL == ctx)
    return GL_TRUE;

  surface = EGL_NO_SURFACE;
  /* Create a p-buffer surface if possible */
  if (pBuffer && createPbufferSurface)
  {
    surface = createPbufferSurface(display, config, pBufferAttribs);
  }
  /* Create a generic surface without a native window, if necessary */
  if (surface==EGL_NO_SURFACE)
  {
    surface = createWindowSurface(display, config, (EGLNativeWindowType) NULL, NULL);
  }
#if 0
  if (surface == EGL_NO_SURFACE)
    return GL_TRUE;
#endif

  if (makeCurrent(display, surface, surface, ctx) != EGL_TRUE)
    return GL_TRUE;

  return GL_FALSE;
}

void glewDestroyContext ()
{
  if (NULL != ctx) eglDestroyContext(display, ctx);
}

#elif defined(GLEW_OSMESA)
OSMesaContext ctx;

static const GLint osmFormat = GL_UNSIGNED_BYTE;
static const GLint osmWidth = 640;
static const GLint osmHeight = 480;
static GLubyte *osmPixels = NULL;

GLboolean glewCreateContext (struct createParams *params)
{
  ctx = OSMesaCreateContext(OSMESA_RGBA, NULL);
  if (NULL == ctx) return GL_TRUE;
  if (NULL == osmPixels)
  {
    osmPixels = (GLubyte *) calloc(osmWidth*osmHeight*4, 1);
  }
  if (!OSMesaMakeCurrent(ctx, osmPixels, GL_UNSIGNED_BYTE, osmWidth, osmHeight))
  {
      return GL_TRUE;
  }
  return GL_FALSE;
}

void glewDestroyContext ()
{
  if (NULL != ctx) OSMesaDestroyContext(ctx);
}

#elif defined(_WIN32)

HWND wnd = NULL;
HDC dc = NULL;
HGLRC rc = NULL;

GLboolean glewCreateContext (struct createParams* params)
{
  WNDCLASS wc;
  PIXELFORMATDESCRIPTOR pfd;
  /* register window class */
  ZeroMemory(&wc, sizeof(WNDCLASS));
  wc.hInstance = GetModuleHandle(NULL);
  wc.lpfnWndProc = DefWindowProc;
  wc.lpszClassName = "GLEW";
  if (0 == RegisterClass(&wc)) return GL_TRUE;
  /* create window */
  wnd = CreateWindow("GLEW", "GLEW", 0, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
                     CW_USEDEFAULT, NULL, NULL, GetModuleHandle(NULL), NULL);
  if (NULL == wnd) return GL_TRUE;
  /* get the device context */
  dc = GetDC(wnd);
  if (NULL == dc) return GL_TRUE;
  /* find pixel format */
  ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));
  if (params->pixelformat == -1) /* find default */
  {
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
    params->pixelformat = ChoosePixelFormat(dc, &pfd);
    if (params->pixelformat == 0) return GL_TRUE;
  }
  /* set the pixel format for the dc */
  if (FALSE == SetPixelFormat(dc, params->pixelformat, &pfd)) return GL_TRUE;
  /* create rendering context */
  rc = wglCreateContext(dc);
  if (NULL == rc) return GL_TRUE;
  if (FALSE == wglMakeCurrent(dc, rc)) return GL_TRUE;
  if (params->major || params->profile || params->flags)
  {
    HGLRC oldRC = rc;
    int contextAttrs[20];
    int i;

    wglewInit();

    /* Intel HD 3000 has WGL_ARB_create_context, but not WGL_ARB_create_context_profile */
    if (!wglewGetExtension("WGL_ARB_create_context"))
      return GL_TRUE;

    i = 0;
    if (params->major)
    {
      contextAttrs[i++] = WGL_CONTEXT_MAJOR_VERSION_ARB;
      contextAttrs[i++] = params->major;
      contextAttrs[i++] = WGL_CONTEXT_MINOR_VERSION_ARB;
      contextAttrs[i++] = params->minor;
    }
    if (params->profile)
    {
      contextAttrs[i++] = WGL_CONTEXT_PROFILE_MASK_ARB;
      contextAttrs[i++] = params->profile;
    }
    if (params->flags)
    {
      contextAttrs[i++] = WGL_CONTEXT_FLAGS_ARB;
      contextAttrs[i++] = params->flags;
    }
    contextAttrs[i++] = 0;
    rc = wglCreateContextAttribsARB(dc, 0, contextAttrs);

    if (NULL == rc) return GL_TRUE;
    if (!wglMakeCurrent(dc, rc)) return GL_TRUE;

    wglDeleteContext(oldRC);
  }
  return GL_FALSE;
}

void glewDestroyContext ()
{
  if (NULL != rc) wglMakeCurrent(NULL, NULL);
  if (NULL != rc) wglDeleteContext(rc);
  if (NULL != wnd && NULL != dc) ReleaseDC(wnd, dc);
  if (NULL != wnd) DestroyWindow(wnd);
  UnregisterClass("GLEW", GetModuleHandle(NULL));
}

/* ------------------------------------------------------------------------ */

#elif defined(__APPLE__) && !defined(GLEW_APPLE_GLX)

#include <OpenGL/OpenGL.h>
#include <OpenGL/CGLTypes.h>

CGLContextObj ctx, octx;

GLboolean glewCreateContext (struct createParams *params)
{
  CGLPixelFormatAttribute contextAttrs[20];
  int i;
  CGLPixelFormatObj pf;
  GLint npix;
  CGLError error;

  i = 0;
  contextAttrs[i++] = kCGLPFAAccelerated; /* No software rendering */

  /* MAC_OS_X_VERSION_10_7  == 1070 */
  #if MAC_OS_X_VERSION_MIN_REQUIRED >= 1070
  if (params->profile & GL_CONTEXT_CORE_PROFILE_BIT)
  {
    if ((params->major==3 && params->minor>=2) || params->major>3)
    {
      contextAttrs[i++] = kCGLPFAOpenGLProfile;                                /* OSX 10.7 Lion onwards */
      contextAttrs[i++] = (CGLPixelFormatAttribute) kCGLOGLPVersion_3_2_Core;  /* 3.2 Core Context      */
    }
  }
  #endif

  contextAttrs[i++] = 0;

  error = CGLChoosePixelFormat(contextAttrs, &pf, &npix);
  if (error) return GL_TRUE;
  error = CGLCreateContext(pf, NULL, &ctx);
  if (error) return GL_TRUE;
  CGLReleasePixelFormat(pf);
  octx = CGLGetCurrentContext();
  error = CGLSetCurrentContext(ctx);
  if (error) return GL_TRUE;
  /* Needed for Regal on the Mac */
  #if defined(GLEW_REGAL) && defined(__APPLE__)
  RegalMakeCurrent(ctx);
  #endif
  return GL_FALSE;
}

void glewDestroyContext ()
{
  CGLSetCurrentContext(octx);
  CGLReleaseContext(ctx);
}

/* ------------------------------------------------------------------------ */

#elif defined(__HAIKU__)

GLboolean glewCreateContext (struct createParams *params)
{
  /* TODO: Haiku: We need to call C++ code here */
  return GL_FALSE;
}

void glewDestroyContext ()
{
  /* TODO: Haiku: We need to call C++ code here */
}

/* ------------------------------------------------------------------------ */

#else /* __UNIX || (__APPLE__ && GLEW_APPLE_GLX) */

Display* dpy = NULL;
XVisualInfo* vi = NULL;
XVisualInfo* vis = NULL;
GLXContext ctx = NULL;
Window wnd = 0;
Colormap cmap = 0;

GLboolean glewCreateContext (struct createParams *params)
{
  int attrib[] = { GLX_RGBA, GLX_DOUBLEBUFFER, None };
  int erb, evb;
  XSetWindowAttributes swa;
  /* open display */
  dpy = XOpenDisplay(params->display);
  if (NULL == dpy) return GL_TRUE;
  /* query for glx */
  if (!glXQueryExtension(dpy, &erb, &evb)) return GL_TRUE;
  /* choose visual */
  if (params->visual == -1)
  {
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attrib);
    if (NULL == vi) return GL_TRUE;
    params->visual = (int)XVisualIDFromVisual(vi->visual);
  }
  else
  {
    int n_vis, i;
    vis = XGetVisualInfo(dpy, 0, NULL, &n_vis);
    for (i=0; i<n_vis; i++)
    {
      if ((int)XVisualIDFromVisual(vis[i].visual) == params->visual)
        vi = &vis[i];
    }
    if (vi == NULL) return GL_TRUE;
  }
  /* create context */
  ctx = glXCreateContext(dpy, vi, None, True);
  if (NULL == ctx) return GL_TRUE;
  /* create window */
  /*wnd = XCreateSimpleWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, 1, 1, 1, 0, 0);*/
  cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
  swa.border_pixel = 0;
  swa.colormap = cmap;
  wnd = XCreateWindow(dpy, RootWindow(dpy, vi->screen),
                      0, 0, 1, 1, 0, vi->depth, InputOutput, vi->visual,
                      CWBorderPixel | CWColormap, &swa);
  /* make context current */
  if (!glXMakeCurrent(dpy, wnd, ctx)) return GL_TRUE;
  if (params->major || params->profile || params->flags)
  {
    GLXContext oldCtx = ctx;
    GLXFBConfig *FBConfigs;
    int FBConfigAttrs[] = { GLX_FBCONFIG_ID, 0, None };
    int contextAttrs[20];
    int nelems, i;

    glxewInit();

    if (!glxewGetExtension("GLX_ARB_create_context"))
      return GL_TRUE;

    if (glXQueryContext(dpy, oldCtx, GLX_FBCONFIG_ID, &FBConfigAttrs[1]))
      return GL_TRUE;
    FBConfigs = glXChooseFBConfig(dpy, vi->screen, FBConfigAttrs, &nelems);

    if (nelems < 1)
      return GL_TRUE;

    i = 0;
    if (params->major)
    {
      contextAttrs[i++] = GLX_CONTEXT_MAJOR_VERSION_ARB;
      contextAttrs[i++] = params->major;
      contextAttrs[i++] = GLX_CONTEXT_MINOR_VERSION_ARB;
      contextAttrs[i++] = params->minor;
    }
    if (params->profile)
    {
      contextAttrs[i++] = GLX_CONTEXT_PROFILE_MASK_ARB;
      contextAttrs[i++] = params->profile;
    }
    if (params->flags)
    {
      contextAttrs[i++] = GLX_CONTEXT_FLAGS_ARB;
      contextAttrs[i++] = params->flags;
    }
    contextAttrs[i++] = None;
    ctx = glXCreateContextAttribsARB(dpy, *FBConfigs, NULL, True, contextAttrs);

    if (NULL == ctx) return GL_TRUE;
    if (!glXMakeCurrent(dpy, wnd, ctx)) return GL_TRUE;

    glXDestroyContext(dpy, oldCtx);

    XFree(FBConfigs);
  }
  return GL_FALSE;
}

void glewDestroyContext ()
{
  if (NULL != dpy && NULL != ctx) glXDestroyContext(dpy, ctx);
  if (NULL != dpy && 0 != wnd) XDestroyWindow(dpy, wnd);
  if (NULL != dpy && 0 != cmap) XFreeColormap(dpy, cmap);
  if (NULL != vis)
    XFree(vis);
  else if (NULL != vi)
    XFree(vi);
  if (NULL != dpy) XCloseDisplay(dpy);
}

#endif /* __UNIX || (__APPLE__ && GLEW_APPLE_GLX) */
