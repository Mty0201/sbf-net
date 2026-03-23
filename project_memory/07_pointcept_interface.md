# Pointcept Interface

- 上游依赖模块: `pointcept.models.builder`。
- 上游依赖模块: `PT-v3m1` backbone。
- 上游依赖模块: `pointcept.models.utils.structure.Point`。
- 上游依赖模块: `pointcept.models.losses.lovasz.LovaszLoss`。
- 上游依赖模块: `pointcept.datasets.defaults.DefaultDataset`。
- 上游依赖模块: `pointcept.datasets.builder.DATASETS` / `build_dataset`。
- 上游依赖模块: `pointcept.datasets.transform.TRANSFORMS` 与配置里使用的标准 transforms。
- 上游依赖模块: `pointcept.datasets.utils.point_collate_fn`。
- 接入方式/model: 项目模型通过 `@MODELS.register_module()` 注册到 Pointcept registry。
- 接入方式/dataset: `BFDataset` 通过 `@DATASETS.register_module()` 接入 Pointcept dataset builder。
- 接入方式/transform: `InjectIndexValidKeys` 通过 `@TRANSFORMS.register_module()` 接入数据 pipeline。
- 接入方式/config: 项目配置直接复用 Pointcept PT-v3m1 backbone 配置和 Pointcept transform 名称。
- 接入方式/runtime: `scripts/train/train.py` 在启动时把本项目根目录和 Pointcept 根目录加入 `sys.path`。
- 接入方式/trainer: 项目 trainer 先导入 `project.datasets / project.models / project.transforms`，再调用 Pointcept 的 `build_dataset` 和 `build_model`。
- 不能修改: Pointcept PT-v3 主干实现与其 registry 协议。
- 不能修改: Pointcept `DefaultDataset` 与 `point_collate_fn` 所要求的数据键结构。
- 不能修改: 模型输入键 `coord`, `grid_coord`, `feat`, `offset` 的接口约定。
- 不能修改: 本项目主线不以重写 Pointcept trainer 或改 Pointcept 源码为实现路径。
