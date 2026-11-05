@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

title Compilador de Archivos
echo.
echo Compilando archivos...
echo.

set "ARCHIVO_SALIDA=compilado.txt"
set "MAX_LINEAS=12000"
set "PARTE=1"
set "ARCHIVO_ACTUAL=compilado%PARTE%.txt"
set "LINEAS_ACT=0"

:: Eliminar archivos de salida anteriores
if exist "compilado*.txt" del "compilado*.txt"

set ARCHIVOS_PROCESADOS=0
set ARCHIVOS_EXCLUIDOS=0

:: Extensiones de archivos a EXCLUIR (no aparecen en el compilado)
set "EXTENSIONES_EXCLUIR=.jpg .jpeg .png .gif .bmp .tiff .tif .webp .svg .ico .raw .psd .ai .eps .mp4 .avi .mov .wmv .flv .mkv .webm .m4v .mpg .mpeg .pdf .otf .ttf .woff .woff2 .eot .db .csv .pyc"

:: Crear primer archivo
echo COMPILADO DE ARCHIVOS > "%ARCHIVO_ACTUAL%"
echo. >> "%ARCHIVO_ACTUAL%"
set "LINEAS_ACT=2"

:: Procesar archivos recursivamente pero excluir node_modules, .git y package-lock.json
for /f "delims=" %%F in ('dir /s /b /a-d * ^| findstr /v /i "\\node_modules\\ \\.output\\  \\src\__pycache__\\ \\mlartifacts\\ \\.git\\ \\dist\\ \\public\\ \\tests\__init__.py\\ \\.nuxt\\ package-lock\.json"') do (
    if not "%%F"=="%~f0" (
        if not "%%F"=="%CD%\compilado*.txt" (
            
            :: Obtener ruta relativa
            set "ruta_completa=%%F"
            set "ruta_relativa=!ruta_completa:%CD%\=!"
            
            :: Verificar si es archivo a excluir (imagen, video, pdf, otf)
            set "excluir=0"
            for %%E in (%EXTENSIONES_EXCLUIR%) do (
                if /i "%%~xF"=="%%E" set "excluir=1"
            )
            
            if !excluir!==0 (
                
                :: Calcular líneas del archivo
                set "lineas_arch=0"
                for /f %%N in ('type "%%F" 2^>nul ^| find /c /v ""') do set "lineas_arch=%%N"
                
                :: Calcular líneas del encabezado
                set /a "encabezado=4"
                set /a "total_necesario=encabezado+lineas_arch"
                set /a "nuevo_total=LINEAS_ACT+total_necesario"
                
                :: Si no cabe, crear nuevo archivo
                if !nuevo_total! GTR %MAX_LINEAS% (
                    set /a PARTE+=1
                    set "ARCHIVO_ACTUAL=compilado!PARTE!.txt"
                    echo COMPILADO DE ARCHIVOS - Parte !PARTE! > "!ARCHIVO_ACTUAL!"
                    echo. >> "!ARCHIVO_ACTUAL!"
                    set "LINEAS_ACT=2"
                )
                
                :: Escribir en el archivo actual
                echo. >> "!ARCHIVO_ACTUAL!"
                echo ================================= >> "!ARCHIVO_ACTUAL!"
                echo !ruta_relativa! >> "!ARCHIVO_ACTUAL!"
                echo ================================= >> "!ARCHIVO_ACTUAL!"
                echo. >> "!ARCHIVO_ACTUAL!"
                type "%%F" >> "!ARCHIVO_ACTUAL!" 2>nul
                
                set /a LINEAS_ACT+=total_necesario
                set /a ARCHIVOS_PROCESADOS+=1
                echo Procesando: !ruta_relativa!
                
            ) else (
                set /a ARCHIVOS_EXCLUIDOS+=1
            )
        )
    )
)

echo. >> "!ARCHIVO_ACTUAL!"
echo Total archivos procesados: %ARCHIVOS_PROCESADOS% >> "!ARCHIVO_ACTUAL%"

echo.
echo Compilacion completada!
echo Archivos procesados: %ARCHIVOS_PROCESADOS%
echo Archivos excluidos: %ARCHIVOS_EXCLUIDOS%
echo Archivos generados: %PARTE%
echo.
pause