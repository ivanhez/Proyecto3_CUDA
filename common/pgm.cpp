#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pgm.h"

PGMImage::PGMImage()
{
    x_dim = 0;
    y_dim = 0;
    pixels = NULL;
}

PGMImage::PGMImage(const char *filename)
{
    pixels = NULL;
    readPGM(filename);
}

PGMImage::~PGMImage()
{
    if (pixels != NULL)
    {
        free(pixels);
        pixels = NULL;
    }
}

void PGMImage::readPGM(const char *filename)
{
    FILE *input = fopen(filename, "rb");
    if (input == NULL)
    {
        printf("Error: Unable to open file '%s'\n", filename);
        exit(1);
    }

    char magic[3];
    int maxval;

    if (fscanf(input, "%2s", magic) != 1)
    {
        printf("Error: Invalid PGM file format\n");
        fclose(input);
        exit(1);
    }

    if (strcmp(magic, "P5") != 0)
    {
        printf("Error: Only P5 (binary) PGM files are supported\n");
        fclose(input);
        exit(1);
    }

    char c = getc(input);
    while (c == '#' || c == '\n' || c == ' ' || c == '\t' || c == '\r')
    {
        if (c == '#')
        {
            while (getc(input) != '\n');
        }
        c = getc(input);
    }
    ungetc(c, input);

    // Read dimensions
    if (fscanf(input, "%d %d", &x_dim, &y_dim) != 2)
    {
        printf("Error: Invalid image dimensions\n");
        fclose(input);
        exit(1);
    }

    // Read max value
    if (fscanf(input, "%d", &maxval) != 1)
    {
        printf("Error: Invalid max value\n");
        fclose(input);
        exit(1);
    }

    if (maxval > 255)
    {
        printf("Error: Only 8-bit PGM files are supported\n");
        fclose(input);
        exit(1);
    }

    // Skip single whitespace character after maxval
    getc(input);

    // Allocate memory for pixel data
    pixels = (unsigned char *)malloc(x_dim * y_dim * sizeof(unsigned char));
    if (pixels == NULL)
    {
        printf("Error: Unable to allocate memory for image\n");
        fclose(input);
        exit(1);
    }

    // Read pixel data
    size_t numRead = fread(pixels, sizeof(unsigned char), x_dim * y_dim, input);
    if (numRead != (size_t)(x_dim * y_dim))
    {
        printf("Error: Unable to read pixel data\n");
        free(pixels);
        fclose(input);
        exit(1);
    }

    fclose(input);
    printf("Successfully loaded PGM image: %s (%d x %d)\n", filename, x_dim, y_dim);
}

// Save image to PGM file
void PGMImage::save(const char *filename)
{
    if (pixels == NULL)
    {
        printf("Error: No image data to save\n");
        return;
    }

    FILE *output = fopen(filename, "wb");
    if (output == NULL)
    {
        printf("Error: Unable to create file '%s'\n", filename);
        return;
    }

    // Write PGM header
    fprintf(output, "P5\n");
    fprintf(output, "# Created by PGMImage\n");
    fprintf(output, "%d %d\n", x_dim, y_dim);
    fprintf(output, "255\n");

    // Write pixel data
    fwrite(pixels, sizeof(unsigned char), x_dim * y_dim, output);

    fclose(output);
    printf("Successfully saved PGM image: %s\n", filename);
}