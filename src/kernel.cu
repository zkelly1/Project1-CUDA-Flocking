#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
int blockSize = 128;

// Part 2.2: use 2.0f for cells twice the maximum interaction distance
// (at most 8 candidate cells), or 1.0f for cells equal to that distance
// (at most 27 candidate cells).
float gridCellWidthMultiplier = 2.0f;

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// Scratch buffers used to reorder particle data into grid-cell order for the
// semi-coherent neighbor search.
glm::vec3 *dev_pos_coherent;
glm::vec3 *dev_vel_coherent;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

void Boids::setBlockSize(int size) {
  blockSize = size;
  threadsPerBlock = dim3(size);
}

void Boids::setGridCellWidthMultiplier(float multiplier) {
  gridCellWidthMultiplier = multiplier;
}

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = gridCellWidthMultiplier
    * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum = glm::vec3(-halfGridWidth);

  // One int per particle stores its original array index after sorting.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  // One int per particle stores the index of its grid cell.
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  // One int per grid cell stores where its particles start in the sorted array.
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  // One int per grid cell stores where its particles end in the sorted array.
  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // One vec3 per particle stores its 3D position in grid-sorted order.
  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  // One vec3 per particle stores its 3D velocity in grid-sorted order.
  cudaMalloc((void**)&dev_vel_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel_coherent failed!");

  dev_thrust_particleArrayIndices =
    thrust::device_pointer_cast(dev_particleArrayIndices);
  dev_thrust_particleGridIndices =
    thrust::device_pointer_cast(dev_particleGridIndices);

  cudaDeviceSynchronize();
  checkCUDAErrorWithLine("initSimulation synchronization failed!");
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/

__device__ glm::vec3 rule1(int N, int iSelf, const glm::vec3 *pos) {
  glm::vec3 perceived_center(0.0f);
  unsigned int number_of_neighbors = 0;

  for (int b_i = 0; b_i < N; b_i++) {
    if(b_i != iSelf && glm::distance(pos[b_i], pos[iSelf]) < rule1Distance) {
      perceived_center += pos[b_i];
      number_of_neighbors++;
    }
  }

  if (number_of_neighbors == 0) {
    return glm::vec3(0.0f);
  }

  perceived_center /= number_of_neighbors;

  return (perceived_center - pos[iSelf]) * rule1Scale;
}

__device__ glm::vec3 rule2(int N, int iSelf, const glm::vec3 *pos) {
  glm::vec3 c(0.0f);

  for (int b_i = 0; b_i < N; b_i++) {
    if(b_i != iSelf && glm::distance(pos[b_i], pos[iSelf]) < rule2Distance) {
      c -= (pos[b_i] - pos[iSelf]);
    }
  }

  return c * rule2Scale;
}

__device__ glm::vec3 rule3(int N, int iSelf, const glm::vec3 *pos,
  const glm::vec3 *vel) {
  glm::vec3 perceived_velocity(0.0f);
  unsigned int number_of_neighbors = 0;

  for (int b_i = 0; b_i < N; b_i++) {
    if(b_i != iSelf && glm::distance(pos[b_i], pos[iSelf]) < rule3Distance) {
      perceived_velocity += vel[b_i];
      number_of_neighbors++;
    }
  }

  if (number_of_neighbors == 0) {
    return glm::vec3(0.0f);
  }

  perceived_velocity /= number_of_neighbors;
  return perceived_velocity * rule3Scale;
}

__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  glm::vec3 perceived_center = rule1(N, iSelf, pos);

  // Rule 2: boids try to stay a distance d away from each other
  glm::vec3 c = rule2(N, iSelf, pos);

  // Rule 3: boids try to match the speed of surrounding boids
  glm::vec3 perceived_velocity = rule3(N, iSelf, pos, vel);

  return perceived_center + c + perceived_velocity;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Compute a new velocity based on pos and vel1
  glm::vec3 new_velocity = vel1[index]
    + computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
  float speed = glm::length(new_velocity);
  if (speed > maxSpeed) {
    new_velocity *= maxSpeed / speed;
  }

  // Record the new velocity into vel2. Question: why NOT vel1?
  vel2[index] = new_velocity;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__device__ glm::ivec3 gridCellForPosition(glm::vec3 position,
  glm::vec3 gridMin, float inverseCellWidth, int gridResolution) {
  glm::ivec3 cell;
  cell.x = (int)floorf((position.x - gridMin.x) * inverseCellWidth);
  cell.y = (int)floorf((position.y - gridMin.y) * inverseCellWidth);
  cell.z = (int)floorf((position.z - gridMin.z) * inverseCellWidth);

  cell.x = imax(0, imin(cell.x, gridResolution - 1));
  cell.y = imax(0, imin(cell.y, gridResolution - 1));
  cell.z = imax(0, imin(cell.z, gridResolution - 1));
  return cell;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  glm::ivec3 cell = gridCellForPosition(
    pos[index], gridMin, inverseCellWidth, gridResolution);
  gridIndices[index] = gridIndex3Dto1D(
    cell.x, cell.y, cell.z, gridResolution);
  indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  int cell = particleGridIndices[index];
  if (index == 0 || particleGridIndices[index - 1] != cell) {
    gridCellStartIndices[cell] = index;
  }
  if (index == N - 1 || particleGridIndices[index + 1] != cell) {
    // Cell ranges are half-open: [start, end).
    gridCellEndIndices[cell] = index + 1;
  }
}

__device__ float maximumRuleDistance() {
  return fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);
}

__device__ void neighborCellBounds(float coordinate, float grid_min,
  float inverse_cell_width, int grid_resolution,
  int *minimum_cell, int *maximum_cell) {
  float radius = maximumRuleDistance();

  // First find the lower and upper bounds in coordinate space, then convert
  // them to grid cell space.
  float lower_bound = coordinate - radius;
  float upper_bound = coordinate + radius;
  int first = (int)floorf((lower_bound - grid_min) * inverse_cell_width);
  int last = (int)ceilf((upper_bound - grid_min) * inverse_cell_width) - 1;

  *minimum_cell = imax(0, first);
  *maximum_cell = imin(grid_resolution - 1, last);
}

__device__ void accumulateNeighborRules(
  glm::vec3 self_position, glm::vec3 neighbor_position,
  glm::vec3 neighbor_velocity,
  glm::vec3 *perceived_center, unsigned int *number_of_neighbors_rule1,
  glm::vec3 *c,
  glm::vec3 *perceived_velocity, unsigned int *number_of_neighbors_rule3) {
  float distance = glm::distance(self_position, neighbor_position);

  if (distance < rule1Distance) {
    *perceived_center += neighbor_position;
    ++(*number_of_neighbors_rule1);
  }
  if (distance < rule2Distance) {
    *c -= neighbor_position - self_position;
  }
  if (distance < rule3Distance) {
    *perceived_velocity += neighbor_velocity;
    ++(*number_of_neighbors_rule3);
  }
}

__device__ glm::vec3 finishVelocityUpdate(glm::vec3 self_position,
  glm::vec3 old_velocity,
  glm::vec3 perceived_center, unsigned int number_of_neighbors_rule1,
  glm::vec3 c,
  glm::vec3 perceived_velocity, unsigned int number_of_neighbors_rule3) {
  glm::vec3 velocity_change(0.0f);

  if (number_of_neighbors_rule1 > 0) {
    perceived_center /= number_of_neighbors_rule1;
    velocity_change += (perceived_center - self_position) * rule1Scale;
  }

  velocity_change += c * rule2Scale;

  if (number_of_neighbors_rule3 > 0) {
    perceived_velocity /= number_of_neighbors_rule3;
    velocity_change += perceived_velocity * rule3Scale;
  }

  glm::vec3 new_velocity = old_velocity + velocity_change;
  float speed = glm::length(new_velocity);
  if (speed > maxSpeed) {
    new_velocity *= maxSpeed / speed;
  }
  return new_velocity;
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  glm::vec3 self_position = pos[index];
  glm::vec3 perceived_center(0.0f);
  glm::vec3 c(0.0f);
  glm::vec3 perceived_velocity(0.0f);
  unsigned int number_of_neighbors_rule1 = 0;
  unsigned int number_of_neighbors_rule3 = 0;

  int min_x, max_x, min_y, max_y, min_z, max_z;
  neighborCellBounds(self_position.x, gridMin.x, inverseCellWidth,
    gridResolution, &min_x, &max_x);
  neighborCellBounds(self_position.y, gridMin.y, inverseCellWidth,
    gridResolution, &min_y, &max_y);
  neighborCellBounds(self_position.z, gridMin.z, inverseCellWidth,
    gridResolution, &min_z, &max_z);

  // x is innermost because adjacent x cells have adjacent 1D indices.
  for (int z = min_z; z <= max_z; ++z) {
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        int cell = gridIndex3Dto1D(x, y, z, gridResolution);
        int start = gridCellStartIndices[cell];
        if (start == -1) {
          continue;
        }

        int end = gridCellEndIndices[cell];
        for (int sorted_index = start; sorted_index < end; ++sorted_index) {
          int neighbor_index = particleArrayIndices[sorted_index];
          if (neighbor_index == index) {
            continue;
          }
          accumulateNeighborRules(self_position, pos[neighbor_index],
            vel1[neighbor_index], &perceived_center,
            &number_of_neighbors_rule1, &c, &perceived_velocity,
            &number_of_neighbors_rule3);
        }
      }
    }
  }

  vel2[index] = finishVelocityUpdate(self_position, vel1[index],
    perceived_center, number_of_neighbors_rule1, c,
    perceived_velocity, number_of_neighbors_rule3);
}

__global__ void kernGatherCoherent(int N, int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel,
  glm::vec3 *coherent_pos, glm::vec3 *coherent_vel) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  int source_index = particleArrayIndices[index];
  coherent_pos[index] = pos[source_index];
  coherent_vel[index] = vel[source_index];
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
    return;
  }

  glm::vec3 self_position = pos[index];
  glm::vec3 perceived_center(0.0f);
  glm::vec3 c(0.0f);
  glm::vec3 perceived_velocity(0.0f);
  unsigned int number_of_neighbors_rule1 = 0;
  unsigned int number_of_neighbors_rule3 = 0;

  int min_x, max_x, min_y, max_y, min_z, max_z;
  neighborCellBounds(self_position.x, gridMin.x, inverseCellWidth,
    gridResolution, &min_x, &max_x);
  neighborCellBounds(self_position.y, gridMin.y, inverseCellWidth,
    gridResolution, &min_y, &max_y);
  neighborCellBounds(self_position.z, gridMin.z, inverseCellWidth,
    gridResolution, &min_z, &max_z);

  for (int z = min_z; z <= max_z; ++z) {
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        int cell = gridIndex3Dto1D(x, y, z, gridResolution);
        int start = gridCellStartIndices[cell];
        if (start == -1) {
          continue;
        }

        int end = gridCellEndIndices[cell];
        for (int neighbor_index = start; neighbor_index < end;
          ++neighbor_index) {
          if (neighbor_index == index) {
            continue;
          }
          accumulateNeighborRules(self_position, pos[neighbor_index],
            vel1[neighbor_index], &perceived_center,
            &number_of_neighbors_rule1, &c, &perceived_velocity,
            &number_of_neighbors_rule3);
        }
      }
    }
  }

  vel2[index] = finishVelocityUpdate(self_position, vel1[index],
    perceived_center, number_of_neighbors_rule1, c,
    perceived_velocity, number_of_neighbors_rule3);
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  dim3 fullBlocksPerGrid(static_cast<int>(
    std::ceil(static_cast<float>(numObjects) / static_cast<float>(blockSize))));

  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, threadsPerBlock>>>(
    numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos<<<fullBlocksPerGrid, threadsPerBlock>>>(
    numObjects, dt, dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // TODO-1.2 ping-pong the velocity buffers
  glm::vec3 *tmp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects,
    gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos,
    dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  thrust::sort_by_key(dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices);
  checkCUDAErrorWithLine("sort_by_key failed!");

  kernResetIntBuffer<<<fullBlocksPerGridCells, threadsPerBlock>>>(gridCellCount,
    dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullBlocksPerGridCells, threadsPerBlock>>>(gridCellCount,
    dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects,
    dev_particleGridIndices, dev_gridCellStartIndices,
    dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, threadsPerBlock>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
    gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

  kernUpdatePos<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects, dt,
    dev_pos, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos scattered failed!");

  glm::vec3 *tmp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);

  kernComputeIndices<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects,
    gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos,
    dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices coherent failed!");

  thrust::sort_by_key(dev_thrust_particleGridIndices,
    dev_thrust_particleGridIndices + numObjects,
    dev_thrust_particleArrayIndices);
  checkCUDAErrorWithLine("sort_by_key coherent failed!");

  kernResetIntBuffer<<<fullBlocksPerGridCells, threadsPerBlock>>>(gridCellCount,
    dev_gridCellStartIndices, -1);
  kernResetIntBuffer<<<fullBlocksPerGridCells, threadsPerBlock>>>(gridCellCount,
    dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer coherent failed!");

  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects,
    dev_particleGridIndices, dev_gridCellStartIndices,
    dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd coherent failed!");

  kernGatherCoherent<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects,
    dev_particleArrayIndices, dev_pos, dev_vel1,
    dev_pos_coherent, dev_vel_coherent);
  checkCUDAErrorWithLine("kernGatherCoherent failed!");

  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, threadsPerBlock>>>(
    numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
    gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
    dev_pos_coherent, dev_vel_coherent, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");

  kernUpdatePos<<<fullBlocksPerGrid, threadsPerBlock>>>(numObjects, dt,
    dev_pos_coherent, dev_vel2);
  checkCUDAErrorWithLine("kernUpdatePos coherent failed!");

  glm::vec3 *tmp = dev_pos;
  dev_pos = dev_pos_coherent;
  dev_pos_coherent = tmp;

  tmp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = tmp;
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel_coherent);
  checkCUDAErrorWithLine("endSimulation cudaFree failed!");
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
