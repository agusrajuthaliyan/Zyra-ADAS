import 'dart:ffi' as ffi;

import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../assets/asset_bootstrap.dart';
import '../constants.dart';
import '../../features/vehicle_select/application/vehicle_profile_notifier.dart';
import '../../features/vehicle_select/data/vehicle_profile.dart';
import 'zyra_engine.dart';

/// Resolves the single shared `DynamicLibrary` handle for libzyra_perception.so.
/// Flutter loads it once per process — all FFI wrappers share this handle.
final Provider<ffi.DynamicLibrary> zyraLibraryProvider =
    Provider<ffi.DynamicLibrary>((Ref ref) {
  return ffi.DynamicLibrary.open('libzyra_perception.so');
});

/// Engine lifecycle: create → load model → warmup → apply profile thresholds
/// → expose [ZyraEngine] to the rest of the app. On dispose, tears the
/// native engine down cleanly.
final AsyncNotifierProvider<ZyraEngineNotifier, ZyraEngine>
    zyraEngineProvider =
    AsyncNotifierProvider<ZyraEngineNotifier, ZyraEngine>(
        ZyraEngineNotifier.new);

class ZyraEngineNotifier extends AsyncNotifier<ZyraEngine> {
  ZyraEngine? _engine;

  @override
  Future<ZyraEngine> build() async {
    ref.onDispose(() {
      _engine?.dispose();
      _engine = null;
    });

    final ffi.DynamicLibrary lib = ref.watch(zyraLibraryProvider);
    final ModelPaths paths = await AssetBootstrap.ensureModelsExtracted();

    final ZyraEngine engine = ZyraEngine.create(lib);
    try {
      // Attempt 1: Vulkan (GPU). On many devices the Vulkan driver initialises
      // fine but NCNN's shader compilation fails on the first real load — code
      // -3 from zyra_engine_load_model means detector_.load() returned false.
      // Retry with CPU so the app stays functional regardless of GPU support.
      bool loaded = false;
      try {
        if (kDebugMode) debugPrint('[Zyra] loading model with Vulkan…');
        engine.loadModel(
          paramPath: paths.paramPath,
          binPath: paths.binPath,
          useVulkan: true,
        );
        loaded = true;
        if (kDebugMode) debugPrint('[Zyra] model loaded (Vulkan)');
      } on ZyraEngineException catch (e) {
        if (kDebugMode) {
          debugPrint('[Zyra] Vulkan load failed ($e) — retrying with CPU');
        }
      }
      if (!loaded) {
        engine.loadModel(
          paramPath: paths.paramPath,
          binPath: paths.binPath,
          useVulkan: false,
        );
        if (kDebugMode) debugPrint('[Zyra] model loaded (CPU fallback)');
      }
      engine.warmup();
    } catch (_) {
      engine.dispose();
      rethrow;
    }

    // Apply the current vehicle profile's class thresholds (Phase 1 scalar
    // struct doesn't tune thresholds yet, so we push the global defaults
    // from constants.dart). When thresholds move onto the profile, replace
    // this block.
    _applyDefaultThresholds(engine);

    // Push profile-specific tuning later phases will layer on. Right now
    // we just listen so threshold changes propagate without a full rebuild.
    ref.listen<AsyncValue<VehicleProfile?>>(vehicleProfileProvider,
        (AsyncValue<VehicleProfile?>? _,
            AsyncValue<VehicleProfile?> next) {
      // Placeholder — no per-profile threshold overrides yet.
      next.whenData((VehicleProfile? p) {
        if (p != null && kDebugMode) {
          debugPrint('[Zyra] engine: vehicle profile=${p.id}');
        }
      });
    });

    _engine = engine;
    return engine;
  }
}

void _applyDefaultThresholds(ZyraEngine engine) {
  kClassThresholds.forEach((String className, double threshold) {
    final int id = kZyraClasses.indexOf(className);
    if (id >= 0) engine.setClassThreshold(id, threshold);
  });
  engine.setNmsIou(kDefaultNmsIou);
}
