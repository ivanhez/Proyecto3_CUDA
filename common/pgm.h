#ifndef PGMIMAGE_H
#define PGMIMAGE_H

class PGMImage
{
public:
    int x_dim;              // Width of the image
    int y_dim;              // Height of the image
    unsigned char *pixels;  // Pixel data array

    // Constructor that loads a PGM image from file
    PGMImage(const char *filename);
    
    // Default constructor
    PGMImage();
    
    // Destructor
    ~PGMImage();
    
    // Method to save image to PGM file
    void save(const char *filename);
    
private:
    // Helper method to read PGM file
    void readPGM(const char *filename);
};

#endif // PGMIMAGE_H