/**
* @file      main.cpp
* @brief     Example Boids flocking simulation for CIS 5650
* @authors   Liam Boone, Kai Ninomiya, Kangning (Gary) Li
* @date      2013-2017
* @copyright University of Pennsylvania
*/

#include "main.hpp"
#include "kernel.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/gtc/matrix_transform.hpp>

// ================
// Configuration
// ================

// LOOK-1.2 - change this to adjust particle count in the simulation
int number_of_boids = 5000;
const float DT = 0.2f;

enum class SimulationMode {
  Naive,
  Scattered,
  Coherent
};

SimulationMode simulation_mode = SimulationMode::Coherent;
bool window_profile = false;
int warmup_frames = 0;
int measured_frames = 0;
int capture_every = 0;
std::string capture_directory;

SimulationMode readSimulationMode(const std::string &name);
const char *simulationModeName();
void stepSelectedSimulation();
int runBenchmark(int argc, char **argv);
bool readWindowProfileArguments(int argc, char **argv);
void saveFramebuffer(const std::string &path);

/**
* C main function.
*/
int main(int argc, char* argv[]) {
  projectName = "5650 CUDA Intro: Boids";

  if (argc > 1 && std::string(argv[1]) == "--benchmark") {
    return runBenchmark(argc, argv);
  }
  if (argc > 1 && std::string(argv[1]) == "--window-profile") {
    if (!readWindowProfileArguments(argc, argv)) {
      return 1;
    }
  }

  if (init(argc, argv)) {
    mainLoop();
    Boids::endSimulation();
    return 0;
  } else {
    return 1;
  }
}

SimulationMode readSimulationMode(const std::string &name) {
  if (name == "naive") {
    return SimulationMode::Naive;
  }
  if (name == "scattered") {
    return SimulationMode::Scattered;
  }
  return SimulationMode::Coherent;
}

const char *simulationModeName() {
  if (simulation_mode == SimulationMode::Naive) {
    return "naive";
  }
  if (simulation_mode == SimulationMode::Scattered) {
    return "scattered";
  }
  return "coherent";
}

void stepSelectedSimulation() {
  if (simulation_mode == SimulationMode::Naive) {
    Boids::stepSimulationNaive(DT);
  }
  else if (simulation_mode == SimulationMode::Scattered) {
    Boids::stepSimulationScatteredGrid(DT);
  }
  else {
    Boids::stepSimulationCoherentGrid(DT);
  }
}

int runBenchmark(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "Usage: cis5650_boids --benchmark "
      << "<naive|scattered|coherent> <boids> <block-size> "
      << "<cell-width-multiplier> <warmup-steps> <measured-steps>"
      << std::endl;
    return 1;
  }

  std::string mode_name = argv[2];
  if (mode_name != "naive" && mode_name != "scattered"
    && mode_name != "coherent") {
    std::cerr << "Unknown simulation mode: " << mode_name << std::endl;
    return 1;
  }

  simulation_mode = readSimulationMode(mode_name);
  number_of_boids = std::stoi(argv[3]);
  int block_size = std::stoi(argv[4]);
  float cell_width_multiplier = std::stof(argv[5]);
  int benchmark_warmup_steps = std::stoi(argv[6]);
  int benchmark_measured_steps = std::stoi(argv[7]);

  if (number_of_boids <= 0 || block_size <= 0 || block_size > 1024
    || cell_width_multiplier <= 0.0f || benchmark_warmup_steps < 0
    || benchmark_measured_steps <= 0) {
    std::cerr << "Invalid benchmark argument." << std::endl;
    return 1;
  }

  Boids::setBlockSize(block_size);
  Boids::setGridCellWidthMultiplier(cell_width_multiplier);
  Boids::initSimulation(number_of_boids);

  for (int i = 0; i < benchmark_warmup_steps; i++) {
    stepSelectedSimulation();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  for (int i = 0; i < benchmark_measured_steps; i++) {
    stepSelectedSimulation();
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  std::cout << simulationModeName() << ","
    << number_of_boids << ","
    << block_size << ","
    << cell_width_multiplier << ","
    << benchmark_measured_steps << ","
    << elapsed_ms / benchmark_measured_steps << ","
    << benchmark_measured_steps * 1000.0f / elapsed_ms << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  Boids::endSimulation();
  return 0;
}

bool readWindowProfileArguments(int argc, char **argv) {
  if (argc != 8 && argc != 10) {
    std::cerr << "Usage: cis5650_boids --window-profile "
      << "<naive|scattered|coherent> <boids> <block-size> "
      << "<cell-width-multiplier> <warmup-frames> <measured-frames> "
      << "[capture-directory capture-every]" << std::endl;
    return false;
  }

  std::string mode_name = argv[2];
  if (mode_name != "naive" && mode_name != "scattered"
    && mode_name != "coherent") {
    std::cerr << "Unknown simulation mode: " << mode_name << std::endl;
    return false;
  }

  simulation_mode = readSimulationMode(mode_name);
  number_of_boids = std::stoi(argv[3]);
  int block_size = std::stoi(argv[4]);
  float cell_width_multiplier = std::stof(argv[5]);
  warmup_frames = std::stoi(argv[6]);
  measured_frames = std::stoi(argv[7]);
  window_profile = true;

  if (argc == 10) {
    capture_directory = argv[8];
    capture_every = std::stoi(argv[9]);
    std::filesystem::create_directories(capture_directory);
  }

  if (number_of_boids <= 0 || block_size <= 0 || block_size > 1024
    || cell_width_multiplier <= 0.0f || warmup_frames < 0
    || measured_frames <= 0 || capture_every < 0) {
    std::cerr << "Invalid window-profile argument." << std::endl;
    return false;
  }

  Boids::setBlockSize(block_size);
  Boids::setGridCellWidthMultiplier(cell_width_multiplier);
  return true;
}

void saveFramebuffer(const std::string &path) {
  std::vector<unsigned char> pixels(width * height * 3);
  std::vector<unsigned char> flipped(width * height * 3);

  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

  int row_size = width * 3;
  for (int y = 0; y < height; y++) {
    std::copy(pixels.begin() + y * row_size,
      pixels.begin() + (y + 1) * row_size,
      flipped.begin() + (height - y - 1) * row_size);
  }

  std::ofstream file(path, std::ios::binary);
  file << "P6\n" << width << " " << height << "\n255\n";
  file.write(reinterpret_cast<const char *>(flipped.data()), flipped.size());
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
* Initialization of CUDA and GLFW.
*/
bool init(int argc, char **argv) {
  // Set window title to "Student Name: [SM 2.0] GPU Name"
  cudaDeviceProp deviceProp;
  int gpuDevice = 0;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (gpuDevice > device_count) {
    std::cout
    << "Error: GPU device number is greater than the number of devices!"
    << " Perhaps a CUDA-capable GPU is not installed?"
    << std::endl;
    return false;
  }
  cudaGetDeviceProperties(&deviceProp, gpuDevice);
  int major = deviceProp.major;
  int minor = deviceProp.minor;

  std::ostringstream ss;
  ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
  deviceName = ss.str();

  // Window setup stuff
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    std::cout
    << "Error: Could not initialize GLFW!"
    << " Perhaps OpenGL 3.3 isn't available?"
    << std::endl;
    return false;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    return false;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);

  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    return false;
  }

  // Initialize drawing state
  initVAO();

  // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
  // change the device ID.
  cudaGLSetGLDevice(0);

  cudaGLRegisterBufferObject(boidVBO_positions);
  cudaGLRegisterBufferObject(boidVBO_velocities);

  // Initialize N-body simulation
  Boids::initSimulation(number_of_boids);

  updateCamera();

  initShaders(program);

  glEnable(GL_DEPTH_TEST);

  return true;
}

void initVAO() {

  std::unique_ptr<GLfloat[]> bodies{ new GLfloat[4 * number_of_boids] };
  std::unique_ptr<GLuint[]> bindices{ new GLuint[number_of_boids] };

  glm::vec4 ul(-1.0, -1.0, 1.0, 1.0);
  glm::vec4 lr(1.0, 1.0, 0.0, 0.0);

  for (int i = 0; i < number_of_boids; i++) {
    bodies[4 * i + 0] = 0.0f;
    bodies[4 * i + 1] = 0.0f;
    bodies[4 * i + 2] = 0.0f;
    bodies[4 * i + 3] = 1.0f;
    bindices[i] = i;
  }


  glGenVertexArrays(1, &boidVAO); // Attach everything needed to draw a particle to this
  glGenBuffers(1, &boidVBO_positions);
  glGenBuffers(1, &boidVBO_velocities);
  glGenBuffers(1, &boidIBO);

  glBindVertexArray(boidVAO);

  // Bind the positions array to the boidVAO by way of the boidVBO_positions
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_positions); // bind the buffer
  glBufferData(GL_ARRAY_BUFFER, 4 * number_of_boids * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW); // transfer data

  glEnableVertexAttribArray(positionLocation);
  glVertexAttribPointer((GLuint)positionLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  // Bind the velocities array to the boidVAO by way of the boidVBO_velocities
  glBindBuffer(GL_ARRAY_BUFFER, boidVBO_velocities);
  glBufferData(GL_ARRAY_BUFFER, 4 * number_of_boids * sizeof(GLfloat), bodies.get(), GL_DYNAMIC_DRAW);
  glEnableVertexAttribArray(velocitiesLocation);
  glVertexAttribPointer((GLuint)velocitiesLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boidIBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, number_of_boids * sizeof(GLuint), bindices.get(), GL_STATIC_DRAW);

  glBindVertexArray(0);
}

void initShaders(GLuint * program) {
  GLint location;

  program[PROG_BOID] = glslUtility::createProgram(
    "shaders/boid.vert.glsl",
    "shaders/boid.geom.glsl",
    "shaders/boid.frag.glsl", attributeLocations, 2);
    glUseProgram(program[PROG_BOID]);

    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_cameraPos")) != -1) {
      glUniform3fv(location, 1, &cameraPosition[0]);
    }
  }

  //====================================
  // Main loop
  //====================================
  void runCUDA() {
    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not
    // use this buffer

    float4 *dptr = NULL;
    float *dptrVertPositions = NULL;
    float *dptrVertVelocities = NULL;

    cudaGLMapBufferObject((void**)&dptrVertPositions, boidVBO_positions);
    cudaGLMapBufferObject((void**)&dptrVertVelocities, boidVBO_velocities);

    stepSelectedSimulation();
    Boids::copyBoidsToVBO(dptrVertPositions, dptrVertVelocities);

    // unmap buffer object
    cudaGLUnmapBufferObject(boidVBO_positions);
    cudaGLUnmapBufferObject(boidVBO_velocities);
  }

  void mainLoop() {
    double fps = 0;
    double timebase = 0;
    int frame = 0;
    int total_profile_frames = 0;
    int completed_profile_frames = 0;
    double profile_start = 0;

    if (!window_profile) {
      Boids::unitTest(); // LOOK-1.2 Example test code for the CUDA setup.
    }

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();

      if (window_profile && total_profile_frames == warmup_frames) {
        profile_start = glfwGetTime();
      }

      frame++;
      double time = glfwGetTime();

      if (time - timebase > 1.0) {
        fps = frame / (time - timebase);
        timebase = time;
        frame = 0;
      }

      runCUDA();

      std::ostringstream ss;
      ss << "[";
      ss.precision(1);
      ss << std::fixed << fps;
      ss << " fps] " << deviceName;
      glfwSetWindowTitle(window, ss.str().c_str());

      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glUseProgram(program[PROG_BOID]);
      glBindVertexArray(boidVAO);
      glPointSize((GLfloat)pointSize);
      glDrawElements(GL_POINTS, number_of_boids, GL_UNSIGNED_INT, 0);
      glPointSize(1.0f);

      glUseProgram(0);
      glBindVertexArray(0);

      if (window_profile && !capture_directory.empty()
        && total_profile_frames >= warmup_frames
        && capture_every > 0
        && completed_profile_frames % capture_every == 0) {
        std::ostringstream filename;
        filename << capture_directory << "/frame_"
          << completed_profile_frames << ".ppm";
        saveFramebuffer(filename.str());
      }

      glfwSwapBuffers(window);

      if (window_profile) {
        if (total_profile_frames >= warmup_frames) {
          completed_profile_frames++;
        }
        total_profile_frames++;

        if (completed_profile_frames >= measured_frames) {
          glFinish();
          double elapsed = glfwGetTime() - profile_start;
          std::cout << simulationModeName() << ","
            << number_of_boids << ","
            << measured_frames << ","
            << elapsed * 1000.0 / measured_frames << ","
            << measured_frames / elapsed << std::endl;
          glfwSetWindowShouldClose(window, GL_TRUE);
        }
      }
    }
    glfwDestroyWindow(window);
    glfwTerminate();
  }


  void errorCallback(int error, const char *description) {
    fprintf(stderr, "error %d: %s\n", error, description);
  }

  void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GL_TRUE);
    }
  }

  void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  }

  void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (leftMousePressed) {
      // compute new camera parameters
      phi += (xpos - lastX) / width;
      theta -= (ypos - lastY) / height;
      theta = std::fmax(0.01f, std::fmin(theta, 3.14f));
      updateCamera();
    }
    else if (rightMousePressed) {
      zoom += (ypos - lastY) / height;
      zoom = std::fmax(0.1f, std::fmin(zoom, 5.0f));
      updateCamera();
    }

	lastX = xpos;
	lastY = ypos;
  }

  void updateCamera() {
    cameraPosition.x = zoom * sin(phi) * sin(theta);
    cameraPosition.z = zoom * cos(theta);
    cameraPosition.y = zoom * cos(phi) * sin(theta);
    cameraPosition += lookAt;

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, lookAt, glm::vec3(0, 0, 1));
    projection = projection * view;

    GLint location;

    glUseProgram(program[PROG_BOID]);
    if ((location = glGetUniformLocation(program[PROG_BOID], "u_projMatrix")) != -1) {
      glUniformMatrix4fv(location, 1, GL_FALSE, &projection[0][0]);
    }
  }
