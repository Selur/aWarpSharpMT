@echo off
rem Local build with Meson: build\aWarpSharpMT.dll for AviSynth+ and VapourSynth (x64, static runtime).
rem Uses meson and ninja from PATH, else the ones of F:\Python4Claude (pip install meson ninja).
rem Compiler: clang-cl from PATH or F:\Python4Claude\llvm\bin (faster code, used for the release), else MSVC.
setlocal
cd /d "%~dp0"
where meson >nul 2>nul || set "PATH=F:\Python4Claude\Scripts;F:\Python4Claude\Lib\site-packages\bin;%PATH%"
where ninja >nul 2>nul || set "PATH=F:\Python4Claude\Scripts;F:\Python4Claude\Lib\site-packages\bin;%PATH%"
where clang-cl >nul 2>nul || if exist F:\Python4Claude\llvm\bin\clang-cl.exe set "PATH=F:\Python4Claude\llvm\bin;%PATH%"
where clang-cl >nul 2>nul && set "CXX=clang-cl"
if not exist build\build.ninja (
  meson setup build --vsenv -Db_vscrt=mt || exit /b 1
)
meson compile -C build || exit /b 1
