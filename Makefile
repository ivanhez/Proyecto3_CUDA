# Targets principales
all: houghGlobal houghConstant houghShared

# Compilar versión con SOLO memoria GLOBAL
houghGlobal: houghBase.cu pgm.o
	@echo "=========================================="
	@echo "Compilando: MEMORIA GLOBAL únicamente"
	@echo "=========================================="
	nvcc houghBase.cu pgm.o -o houghGlobal
	@echo "✓ Ejecutable creado: houghGlobal"
	@echo ""

# Compilar versión con memoria GLOBAL + CONSTANTE
houghConstant: houghBase.cu pgm.o
	@echo "=========================================="
	@echo "Compilando: MEMORIA GLOBAL + CONSTANTE"
	@echo "=========================================="
	nvcc -DUSE_CONSTANT_MEMORY houghBase.cu pgm.o -o houghConstant
	@echo "✓ Ejecutable creado: houghConstant"
	@echo ""

# Compilar versión con memoria GLOBAL + CONSTANTE + COMPARTIDA
houghShared: houghBase.cu pgm.o
	@echo "=========================================="
	@echo "Compilando: GLOBAL + CONSTANTE + COMPARTIDA"
	@echo "=========================================="
	nvcc -DUSE_SHARED_MEMORY houghBase.cu pgm.o -o houghShared
	@echo "✓ Ejecutable creado: houghShared"
	@echo ""

# Compilar el módulo PGM
pgm.o: common/pgm.cpp
	@echo "Compilando módulo PGM..."
	g++ -c common/pgm.cpp -o ./pgm.o
	@echo "✓ pgm.o creado"
	@echo ""

# Limpiar archivos compilados
clean:
	@echo "Limpiando archivos compilados..."
	rm -f houghGlobal houghConstant houghShared pgm.o output_lines*.png
	@echo "✓ Limpieza completa"

# Limpiar solo las imágenes de salida
clean-output:
	@echo "Eliminando imágenes de salida..."
	rm -f output_lines*.png
	@echo "✓ Imágenes eliminadas"

# Compilar solo versión global
global: houghGlobal

# Compilar solo versión constante
constant: houghConstant

# Compilar solo versión compartida
shared: houghShared

# Ayuda
help:
	@echo "=============================================="
	@echo "Makefile - Transformada de Hough con CUDA"
	@echo "=============================================="
	@echo ""
	@echo "COMPILACIÓN:"
	@echo "  make              - Compila las 3 versiones"
	@echo "  make global       - Solo versión con Memoria Global"
	@echo "  make constant     - Solo versión Global + Constante"
	@echo "  make shared       - Solo versión Global + Constante + Compartida"
	@echo ""
	@echo "LIMPIEZA:"
	@echo "  make clean        - Elimina ejecutables y salidas"
	@echo "  make clean-output - Solo elimina imágenes PNG"
	@echo ""
	@echo "PRUEBAS:"
	@echo "  make test IMG=imagen.pgm [THRESHOLD=valor]"
	@echo "                    - Ejecuta las 3 versiones y compara"
	@echo ""
	@echo "EJECUCIÓN MANUAL:"
	@echo "  ./houghGlobal imagen.pgm [threshold]"
	@echo "  ./houghConstant imagen.pgm [threshold]"
	@echo "  ./houghShared imagen.pgm [threshold]"
	@echo ""
	@echo "EJEMPLO PARA BITÁCORA (10 mediciones):"
	@echo "  for i in {1..10}; do ./houghGlobal imagen.pgm; done"
	@echo "  for i in {1..10}; do ./houghConstant imagen.pgm; done"
	@echo "  for i in {1..10}; do ./houghShared imagen.pgm; done"
	@echo ""

.PHONY: all clean clean-output help global constant shared test