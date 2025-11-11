# Targets principales
all: houghGlobal houghConstant

# Compilar versión con memoria GLOBAL
houghGlobal: houghBase.cu pgm.o
	@echo "Compilando versión con MEMORIA GLOBAL..."
	nvcc houghBase.cu pgm.o -o houghGlobal
	@echo "Ejecutable creado: houghGlobal"

# Compilar versión con memoria CONSTANTE
houghConstant: houghBase.cu pgm.o
	@echo "Compilando versión con MEMORIA CONSTANTE..."
	nvcc -DUSE_CONSTANT_MEMORY houghBase.cu pgm.o -o houghConstant
	@echo "Ejecutable creado: houghConstant"

# Compilar el módulo PGM
pgm.o: common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o

# Limpiar archivos compilados
clean:
	rm -f houghGlobal houghConstant pgm.o output_lines*.png

# Limpiar solo las imágenes de salida
clean-output:
	rm -f output_lines*.png

# Compilar solo versión global
global: houghGlobal

# Compilar solo versión constante
constant: houghConstant

# Ayuda
help:
	@echo "Uso del Makefile:"
	@echo "  make           - Compila ambas versiones (Global y Constante)"
	@echo "  make global    - Compila solo versión con Memoria Global"
	@echo "  make constant  - Compila solo versión con Memoria Constante"
	@echo "  make clean     - Elimina todos los archivos compilados y salidas"
	@echo "  make clean-output - Elimina solo las imágenes de salida"
	@echo ""
	@echo "Ejecutar:"
	@echo "  ./houghGlobal <imagen.pgm> [threshold]"
	@echo "  ./houghConstant <imagen.pgm> [threshold]"

.PHONY: all clean clean-output help global constant