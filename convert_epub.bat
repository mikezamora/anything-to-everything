@echo off
REM Batch script for converting EPUB to audiobook on Windows
REM Usage: convert_epub.bat <epub_file> <speaker_audio> [output_name]

setlocal EnableDelayedExpansion

echo ========================================
echo EPUB to Audiobook Converter
echo ========================================
echo.

REM Check arguments
if "%~1"=="" (
    echo Usage: convert_epub.bat ^<epub_file^> ^<speaker_audio^> [output_name]
    echo.
    echo Example: convert_epub.bat mybook.epub myspeaker.wav
    echo          convert_epub.bat mybook.epub myspeaker.wav myaudiobook.wav
    exit /b 1
)

if "%~2"=="" (
    echo Error: Speaker audio file is required
    echo Usage: convert_epub.bat ^<epub_file^> ^<speaker_audio^> [output_name]
    exit /b 1
)

set EPUB_FILE=%~1
set SPEAKER_AUDIO=%~2

REM Set output filename
if "%~3"=="" (
    set OUTPUT_FILE=audiobook.wav
) else (
    set OUTPUT_FILE=%~3
)

REM Check if files exist
if not exist "%EPUB_FILE%" (
    echo Error: EPUB file not found: %EPUB_FILE%
    exit /b 1
)

if not exist "%SPEAKER_AUDIO%" (
    echo Error: Speaker audio file not found: %SPEAKER_AUDIO%
    exit /b 1
)

echo Input EPUB: %EPUB_FILE%
echo Speaker Audio: %SPEAKER_AUDIO%
echo Output: %OUTPUT_FILE%
echo.

REM Run the converter
python main.py "%EPUB_FILE%" "%SPEAKER_AUDIO%" -o "%OUTPUT_FILE%" --use-fp16 --verbose

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Conversion complete!
    echo Audiobook saved to: %OUTPUT_FILE%
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Conversion failed!
    echo Check the errors above.
    echo ========================================
)

endlocal
