project('aWarpSharpMT', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++17'],
  meson_version : '>=0.64.0',
  version : '2.3.0'
)

sources = [
  'src/core/awarp.cpp',
  'src/core/cpu.cpp',
  'src/core/kernels_c.cpp',
  'src/avs_plugin.cpp',
  'src/vs_plugin.cpp',
]

# Vendored AviSynth+ (3.7.5) and VapourSynth (API 4) headers.
incs = include_directories('include/avisynth', 'include/vapoursynth', 'src')

cxx = meson.get_compiler('cpp')
msvc = cxx.get_argument_syntax() == 'msvc'

if msvc
  add_project_arguments('/D_CRT_SECURE_NO_WARNINGS', language : 'cpp')
else
  add_project_arguments('-Wall', '-Wextra', '-ffp-contract=off', language : 'cpp')
endif

# SIMD kernels, each in its own library with its own instruction set flags; selected at runtime (cpu.cpp).
cpu = host_machine.cpu_family()
simd_libs = []
if cpu == 'x86_64'
  simd = {
    'sse41' : {'src' : 'src/core/kernels_sse41.cpp', 'msvc' : [], 'gcc' : ['-msse4.1']},
    'avx2' : {'src' : 'src/core/kernels_avx2.cpp', 'msvc' : ['/arch:AVX2'], 'gcc' : ['-mavx2']},
  }
  foreach name, s : simd
    simd_libs += static_library('kernels_' + name, s['src'],
      include_directories : incs,
      cpp_args : msvc ? s['msvc'] : s['gcc'],
      gnu_symbol_visibility : 'hidden',
      pic : true
    )
  endforeach
elif cpu == 'aarch64'
  sources += 'src/core/kernels_neon.cpp'
endif

if host_machine.system() == 'windows'
  windows = import('windows')
  sources += windows.compile_resources('src/awarpsharpmt.rc', include_directories : incs)
endif

# VapourSynth only autoloads *.dylib on macOS, while Meson defaults to *.so for modules.
module_kwargs = {}
if host_machine.system() == 'darwin'
  module_kwargs += {'name_suffix' : 'dylib'}
endif

if get_option('python_package')
  # Same layout as the VapourSynth PyPI package autoloads from: site-packages/vapoursynth/plugins
  py = import('python').find_installation(pure : false)
  install_dir = join_paths(py.get_install_dir(), 'vapoursynth', 'plugins')
else
  install_dir = join_paths(get_option('prefix'), get_option('libdir'), 'vapoursynth')
endif

shared_module('aWarpSharpMT', sources,
  include_directories : incs,
  link_with : simd_libs,
  install : true,
  install_dir : install_dir,
  gnu_symbol_visibility : 'hidden',
  kwargs : module_kwargs
)

# Every SIMD tier against the C reference on random data (meson test); not part of wheel builds.
if not get_option('python_package')
  kernel_sources = ['test/kernel_test.cpp', 'src/core/cpu.cpp', 'src/core/kernels_c.cpp']
  if cpu == 'aarch64'
    kernel_sources += 'src/core/kernels_neon.cpp'
  endif
  kernel_test = executable('kernel_test', kernel_sources,
    include_directories : incs,
    link_with : simd_libs,
    install : false
  )
  test('kernels', kernel_test, timeout : 300)
endif
