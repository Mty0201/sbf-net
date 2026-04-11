"""ZAHA offline preprocessing package.

Phase 1 of workstream dataset-handover-s3dis-chas. Turns raw ASCII PCD files from
``/home/mty0201/data/ZAHA_pcd/`` into per-chunk ``coord/segment/normal.npy``
arrays under ``/home/mty0201/data/ZAHA_chunked/`` with a provenance manifest.

See ``data_pre/zaha/docs/README.md`` for package layout and the open3d import
order pitfall warning (``import open3d as o3d`` MUST precede
``pandas``/``scipy``/``sklearn`` in every script to avoid a GLIBCXX conflict in
the ``ptv3`` conda env).
"""
