#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "common/stb_image_write.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 50;
const float radInc = degreeInc * M_PI / 180;

//*****************************************************************
// Declaración de memoria constante (scope global)
#if defined(USE_CONSTANT_MEMORY) || defined(USE_SHARED_MEMORY)
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];
#endif

//*****************************************************************
// Estructura para almacenar información de una línea detectada
typedef struct {
    int rIdx;
    int tIdx;
    int votes;
} DetectedLine;

//*****************************************************************
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset (*acc, 0, sizeof (int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      {
        int idx = j * w + i;
        if (pic[idx] > 0)
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;
            float theta = 0;
            for (int tIdx = 0; tIdx < degreeBins; tIdx++)
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++;
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// Función para convertir imagen B&N a RGB
unsigned char* convertToRGB(unsigned char *grayImg, int w, int h)
{
    unsigned char *rgbImg = (unsigned char *)malloc(w * h * 3);
    for (int i = 0; i < w * h; i++)
    {
        rgbImg[i * 3 + 0] = grayImg[i]; // R
        rgbImg[i * 3 + 1] = grayImg[i]; // G
        rgbImg[i * 3 + 2] = grayImg[i]; // B
    }
    return rgbImg;
}

//*****************************************************************
// Función para dibujar una línea usando el algoritmo de Bresenham
void drawLine(unsigned char *img, int w, int h, int x0, int y0, int x1, int y1, 
              unsigned char r, unsigned char g, unsigned char b)
{
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;

    while (1)
    {
        // Verificar límites
        if (x0 >= 0 && x0 < w && y0 >= 0 && y0 < h)
        {
            int idx = (y0 * w + x0) * 3;
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx)
        {
            err += dx;
            y0 += sy;
        }
    }
}

//*****************************************************************
// Función para dibujar las líneas detectadas sobre la imagen
void drawDetectedLines(unsigned char *img, int w, int h, int *acc, 
                       float rMax, float rScale, int threshold)
{
    int xCent = w / 2;
    int yCent = h / 2;
    int linesDrawn = 0;

    printf("\nDibujando líneas con threshold = %d votos\n", threshold);

    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            int votes = acc[rIdx * degreeBins + tIdx];
            
            if (votes > threshold)
            {
                // Calcular parámetros de la línea
                float theta = tIdx * radInc;
                float r = (rIdx * rScale) - rMax;
                
                // Calcular dos puntos en la línea para dibujarla
                int x0, y0, x1, y1;
                
                if (fabs(sin(theta)) > 0.001) // Línea no vertical
                {
                    x0 = 0;
                    y0 = yCent - (int)((r - (x0 - xCent) * cos(theta)) / sin(theta));
                    x1 = w - 1;
                    y1 = yCent - (int)((r - (x1 - xCent) * cos(theta)) / sin(theta));
                }
                else // Línea vertical
                {
                    x0 = x1 = xCent + (int)(r / cos(theta));
                    y0 = 0;
                    y1 = h - 1;
                }
                
                // Dibujar línea en color (rojo)
                drawLine(img, w, h, x0, y0, x1, y1, 255, 0, 0);
                linesDrawn++;
            }
        }
    }
    
    printf("Total de líneas dibujadas: %d\n", linesDrawn);
}

//*****************************************************************
// Función para calcular threshold automático
int calculateThreshold(int *acc, int size)
{
    long long sum = 0;
    for (int i = 0; i < size; i++)
        sum += acc[i];
    float mean = (float)sum / size;
    
    float variance = 0;
    for (int i = 0; i < size; i++)
    {
        float diff = acc[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    float stddev = sqrt(variance);
    
    int threshold = (int)(mean + 2 * stddev);
    printf("Estadísticas del acumulador:\n");
    printf("  Promedio: %.2f\n", mean);
    printf("  Desviación estándar: %.2f\n", stddev);
    printf("  Threshold (media + 2*stddev): %d\n", threshold);
    
    return threshold;
}

//*****************************************************************
// Kernel con MEMORIA COMPARTIDA
#ifdef USE_SHARED_MEMORY
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, 
                               float rMax, float rScale)
{
  // Definir locID usando IDs de los hilos del bloque
  int locID = threadIdx.x;
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Definir acumulador local en memoria compartida
  __shared__ int localAcc[degreeBins * rBins];
  
  // Inicializar a 0 el acumulador local
  // Cada hilo inicializa múltiples elementos
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
  {
    localAcc[i] = 0;
  }
  
  // Barrera para asegurar que la inicialización está completa
  __syncthreads();
  
  if (gloID < w * h)
  {
    int xCent = w / 2;
    int yCent = h / 2;
    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
      {
        float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
        int rIdx = (r + rMax) / rScale;
        
        // Usar acumulador local con operación atómica
        atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
      }
    }
  }
  
  // Barrera para asegurar que todos completaron el incremento
  __syncthreads();
  
  // Sumar acumulador local al global
  for (int i = locID; i < degreeBins * rBins; i += blockDim.x)
  {
    atomicAdd(&acc[i], localAcc[i]);
  }
}

//*****************************************************************
// Kernel con MEMORIA CONSTANTE (sin compartida)
#elif defined(USE_CONSTANT_MEMORY)
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, 
                               float rMax, float rScale)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}

//*****************************************************************
// Kernel con MEMORIA GLOBAL únicamente
#else
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, 
                               float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gloID >= w * h) return;

  int xCent = w / 2;
  int yCent = h / 2;
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
}
#endif

//*****************************************************************
int main (int argc, char **argv)
{
  if (argc < 2)
  {
      printf("Uso: %s <imagen.pgm> [threshold opcional]\n", argv[0]);
      return 1;
  }

  printf("==============================================\n");
  #ifdef USE_SHARED_MEMORY
  printf("MODO: Usando MEMORIA GLOBAL + CONSTANTE + COMPARTIDA\n");
  #elif defined(USE_CONSTANT_MEMORY)
  printf("MODO: Usando MEMORIA GLOBAL + CONSTANTE\n");
  #else
  printf("MODO: Usando MEMORIA GLOBAL únicamente\n");
  #endif
  printf("==============================================\n\n");

  int i;
  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // Variables para memoria global
  #if !defined(USE_CONSTANT_MEMORY) && !defined(USE_SHARED_MEMORY)
  float* d_Cos;
  float* d_Sin;
  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);
  #endif

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // Pre-calcular valores trigonométricos
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Copiar valores trigonométricos al device
  #if defined(USE_CONSTANT_MEMORY) || defined(USE_SHARED_MEMORY)
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof (float) * degreeBins);
  printf("Valores trigonométricos copiados a MEMORIA CONSTANTE\n");
  #else
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  printf("Valores trigonométricos copiados a MEMORIA GLOBAL\n");
  #endif

  // Setup y copia de datos host->device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;
  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // Configuración de ejecución
  int blockNum = ceil (w * h / 256.0);
  
  // Crear eventos CUDA para medir tiempo
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  // Registrar tiempo de inicio
  cudaEventRecord(start);
  
  // Llamada al kernel según el modo
  #if defined(USE_CONSTANT_MEMORY) || defined(USE_SHARED_MEMORY)
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);
  #else
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  #endif
  
  // Registrar tiempo de fin
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  
  // Calcular tiempo transcurrido
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  #ifdef USE_SHARED_MEMORY
  printf("\nKernel execution time (Global + Constant + Shared Memory): %f ms\n", milliseconds);
  #elif defined(USE_CONSTANT_MEMORY)
  printf("\nKernel execution time (Global + Constant Memory): %f ms\n", milliseconds);
  #else
  printf("\nKernel execution time (Global Memory only): %f ms\n", milliseconds);
  #endif
  
  // Destruir eventos
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Obtener resultados del device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // Comparar resultados CPU vs GPU
  int mismatches = 0;
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
    {
      mismatches++;
      if (mismatches <= 5)
        printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
  }
  if (mismatches > 5)
    printf("... y %d diferencias más\n", mismatches - 5);
  
  printf("Done! Total mismatches: %d / %d (%.2f%%)\n", 
         mismatches, degreeBins * rBins, 
         100.0 * mismatches / (degreeBins * rBins));

  // Generar imagen con líneas detectadas
  printf("\n=== Generando imagen con líneas detectadas ===\n");
  
  int threshold;
  if (argc >= 3)
  {
      threshold = atoi(argv[2]);
      printf("Usando threshold manual: %d\n", threshold);
  }
  else
  {
      threshold = calculateThreshold(h_hough, degreeBins * rBins);
  }
  
  unsigned char *rgbImg = convertToRGB(inImg.pixels, w, h);
  drawDetectedLines(rgbImg, w, h, h_hough, rMax, rScale, threshold);
  
  // Nombre de archivo según el modo
  const char *outputFile;
  #ifdef USE_SHARED_MEMORY
  outputFile = "output_lines_shared.png";
  #elif defined(USE_CONSTANT_MEMORY)
  outputFile = "output_lines_constant.png";
  #else
  outputFile = "output_lines_global.png";
  #endif
  
  if (stbi_write_png(outputFile, w, h, 3, rgbImg, w * 3))
  {
      printf("Imagen guardada exitosamente: %s\n", outputFile);
  }
  else
  {
      printf("Error al guardar la imagen\n");
  }
  
  free(rgbImg);

  // Limpieza de memoria
  cudaFree(d_in);
  cudaFree(d_hough);
  
  #if !defined(USE_CONSTANT_MEMORY) && !defined(USE_SHARED_MEMORY)
  cudaFree(d_Cos);
  cudaFree(d_Sin);
  #endif

  free(h_hough);
  free(pcCos);
  free(pcSin);
  delete[] cpuht;

  cudaDeviceReset();

  return 0;
}
