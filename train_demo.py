"""
EANwH 训练（脸+手 early fusion）。

当前配置：
- 训练/验证使用 CSV 中的 train/val
- 仅训练 A/N/J/S 四类（自动跳过 D/F/W）
- 支持从 checkpoint 恢复
- train/val 不再逐 epoch 保存矩阵
- 按 val_acc 维护 Top-5 checkpoints
- Dial/Solo 可按间隔评测并保存矩阵与报告
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.baseline_data import (
    SequenceFrameDatasetEANwH,
    collect_sequence_entries_eanwh,
    diagnose_eanwh_no_entries,
    filter_csv_with_face_and_hand,
)
from functions.baseline_torch_model import EANwHClassifier

LABEL_NAMES = ["A", "N", "J", "S"]  # 与 baseline_data.LABEL_MAP 对应


def describe_dropped_entries_by_label(entries, keep_labels):
    """
    返回被 keep_labels 过滤掉的样本信息，便于定位“少了哪条”。
    """
    dropped = []
    for e in entries:
        if e["label"] in keep_labels:
            continue
        clip_name = ""
        fps = e.get("face_paths") or []
        if fps:
            clip_name = Path(fps[0]).stem
        dropped.append({
            "label": e["label"],
            "stem": e.get("stem", ""),
            "video_key": e.get("video_key", ""),
            "clip_hint": clip_name,
        })
    return dropped


def evaluate(model, loader, device, return_f1=False):
    """段级别：返回 acc，可选返回 f1。"""
    model.eval()
    total_correct, total = 0, 0
    all_preds, all_golds = [], []
    with torch.no_grad():
        for faces, hands, labels in loader:
            faces, hands = faces.to(device), hands.to(device)
            labels = labels.to(device)
            logits = model(faces, hands)
            preds = logits.argmax(1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_golds.extend(labels.cpu().tolist())
    acc = total_correct / max(total, 1)
    if return_f1:
        f1 = metrics.f1_score(all_golds, all_preds, average="weighted") if all_golds else 0.0
        return acc, f1, all_golds, all_preds
    return acc


def filter_entries_by_labels(entries, keep_labels):
    """仅保留 keep_labels 指定的类别，并重映射为 [0..K-1]。"""
    old_to_new = {old: new for new, old in enumerate(keep_labels)}
    out = []
    for e in entries:
        old_label = e["label"]
        if old_label not in old_to_new:
            continue
        ne = dict(e)
        ne["label"] = old_to_new[old_label]
        out.append(ne)
    return out


def print_and_save_classification_report(
    all_golds, all_preds, save_dir: Path, prefix: str, epoch_idx: int, acc: float, f1: float
):
    """打印并保存混淆矩阵 + classification_report（与 dial 侧一致，行=true 列=pred，顺序 A,N,J,S）。"""
    labels_idx = list(range(len(LABEL_NAMES)))
    cm = metrics.confusion_matrix(all_golds, all_preds, labels=labels_idx)
    report = metrics.classification_report(
        all_golds,
        all_preds,
        labels=labels_idx,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / f"{prefix}_classification_report.txt"
    cm_npy_path = save_dir / f"{prefix}_confusion_matrix.npy"
    cm_txt_path = save_dir / f"{prefix}_confusion_matrix.txt"
    np.save(cm_npy_path, cm)
    with open(cm_txt_path, "w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(LABEL_NAMES) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"epoch={epoch_idx}\n")
        f.write(f"accuracy={acc:.6f}\n")
        f.write(f"f1_weighted={f1:.6f}\n\n")
        f.write("confusion_matrix (rows=true, cols=pred):\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
        f.write("\n")
        f.write(report)
    print(f"[{prefix}] confusion matrix labels: {LABEL_NAMES}")
    print(cm)
    print(f"[{prefix}] Classification Report (precision, recall, f1-score, support):")
    print(report)
    print(f"[{prefix}] saved: {report_path}, {cm_npy_path}, {cm_txt_path}")


def save_topk_by_val_acc(
    model,
    optimizer,
    epoch: int,
    val_acc: float,
    args,
    save_dir: Path,
    topk_records,
    k: int = 5,
):
    """
    维护 val_acc Top-K checkpoints：
    - 始终只保留前 K 个
    - 其余自动删除
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    should_save = len(topk_records) < k or val_acc > min(r["val_acc"] for r in topk_records)
    if not should_save:
        return topk_records

    ckpt_name = f"valtop_epoch{epoch:03d}_acc{val_acc:.6f}.pt"
    ckpt_path = save_dir / ckpt_name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": float(val_acc),
            "args": vars(args),
        },
        ckpt_path,
    )
    topk_records.append({"path": ckpt_path, "val_acc": float(val_acc), "epoch": int(epoch)})
    topk_records.sort(key=lambda x: (x["val_acc"], x["epoch"]), reverse=True)

    if len(topk_records) > k:
        for rec in topk_records[k:]:
            try:
                rec["path"].unlink(missing_ok=True)
            except Exception:
                pass
        topk_records = topk_records[:k]

    print(f"[Save TopK] {ckpt_path}  val_acc={val_acc:.4f}")
    return topk_records


def main():
    p = argparse.ArgumentParser(description="EANwH 训练（脸+手 early fusion）")
    p.add_argument("--face_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_solo/face")
    p.add_argument("--frame_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_solo/frame",
                   help="用于对齐的完整帧目录（与 face 同名）；手部特征优先从 hand_feat_dir 加载")
    p.add_argument("--hand_feat_dir", type=str, default="",
                   help="预抽取手部特征目录（先运行 extract_hand_features.py）。为空则现场跑 MediaPipe（慢）")
    p.add_argument("--train_csv", type=str, default="/home/lr/wangyi/Sign/RO-MAN/eJSL_solo_dataset/train.csv")
    p.add_argument("--val_csv", type=str, default="/home/lr/wangyi/Sign/RO-MAN/eJSL_solo_dataset/val.csv")
    p.add_argument("--test_csv", type=str, default="/home/lr/wangyi/Sign/RO-MAN/eJSL_solo_dataset/test.csv")
    p.add_argument("--dial_face_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/face")
    p.add_argument("--dial_frame_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/frame")
    p.add_argument(
        "--dial_hand_feat_dir",
        type=str,
        default="/raid_zoe/home/lr/wangyi/sign/eJSL_dial/hand_features",
        help="dial 测试集 hand npy 根目录；勿与 BOBSL 的 --hand_feat_dir 混用（以前空串会误回退到 BOBSL）",
    )
    p.add_argument("--dial_test_csv", type=str, default="/home/lr/wangyi/Sign/RO-MAN/eJSL_dial_dataset/test.csv")
    p.add_argument("--updated_csv_dir", type=str, default="",
                   help="若提供，先过滤 face/hand 可用性并写入 *_updated.csv，再用更新后 CSV")
    p.add_argument("--reuse_updated_csv", action="store_true",
                   help="若 *_updated.csv 已存在则直接复用，跳过再次过滤（推荐二次运行开启）")
    p.add_argument("--reuse_entries_cache", action="store_true",
                   help="复用已缓存的 entries（按 csv+win+step 命名）以跳过 Collect 阶段")
    p.add_argument("--save_dir", type=str, default="outputs_eanwh")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--enable_solo_eval", action="store_true", help="开启 solo test（默认关闭，仅跑 dial）")
    p.add_argument("--solo_eval_every", type=int, default=5, help="solo test 每 N 个 epoch 跑一次（仅在开启 solo 时生效）")
    p.add_argument("--dial_eval_every", type=int, default=5, help="dial test 每 N 个 epoch 跑一次")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--win", type=int, default=10, help="每个 clip 采样后的固定帧长")
    p.add_argument("--step", type=int, default=5, help="保留参数（兼容旧脚本），当前不再用于滑窗切段")
    p.add_argument("--hand_stride", type=int, default=1,
                   help="hand npy 期望步长：校验 npy_rows == ceil(n_frame/hand_stride)")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--backbone_pretrain", type=str, default="vggface2", choices=["vggface2", "imagenet", "none"],
                   help="人脸 ResNet50 预训练来源，论文设定为 vggface2")
    p.add_argument("--vggface2_ckpt", type=str, default="",
                   help="当 backbone_pretrain=vggface2 时，提供 PyTorch 权重路径")
    p.add_argument(
        "--resume_path",
        type=str,
        default="/home/lr/wangyi/Sign/RO-MAN/EmoAffectNet/EMO-AffectNetModel-pretrain/outputs_eanwh/best.pt",
        help="从该 checkpoint 恢复继续训练（默认你的 best.pt）",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    def _updated_path(csv_path: str) -> str:
        if not args.updated_csv_dir:
            return csv_path
        stem = Path(csv_path).stem
        if stem.endswith("_updated"):
            stem = stem[: -len("_updated")]
        return str(Path(args.updated_csv_dir) / f"{stem}_updated.csv")

    def _resolve_csv(csv_path: str) -> str:
        if not args.updated_csv_dir:
            return csv_path
        up = Path(_updated_path(csv_path))
        if args.reuse_updated_csv and up.is_file():
            print(f"[CSV Filter] 复用已存在 updated csv: {up}")
            return str(up)
        hand_check_root = args.hand_feat_dir or ""
        return filter_csv_with_face_and_hand(
            csv_path=csv_path,
            face_root=args.face_root,
            hand_root=hand_check_root,
            frame_root=args.frame_root,
            hand_stride=args.hand_stride,
            output_csv=str(up),
        )

    args.train_csv = _resolve_csv(args.train_csv)
    args.val_csv = _resolve_csv(args.val_csv)
    if args.enable_solo_eval:
        args.test_csv = _resolve_csv(args.test_csv)

    cache_dir = Path(args.updated_csv_dir) if args.updated_csv_dir else Path(args.save_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _entries_cache_path(csv_path: str, tag: str) -> Path:
        stem = Path(csv_path).stem
        # v2: 每个 clip 仅 1 条样本（不再滑窗切段），避免复用到旧缓存
        return cache_dir / f"{tag}_{stem}_clip1_w{args.win}_v3.pkl"

    def _collect_or_load_entries(csv_path: str, face_root: str, frame_root: str, tag: str):
        cp = _entries_cache_path(csv_path, tag)
        if args.reuse_entries_cache and cp.is_file():
            with open(cp, "rb") as f:
                entries = pickle.load(f)
            print(f"[Stage] Reuse entries cache: {cp} (n={len(entries)})")
            return entries
        print(f"[Stage] Collect {tag} entries ...")
        entries = collect_sequence_entries_eanwh(
            csv_path, face_root, frame_root, win=args.win, step=args.step
        )
        with open(cp, "wb") as f:
            pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Stage] Saved entries cache: {cp} (n={len(entries)})")
        return entries

    train_entries = _collect_or_load_entries(args.train_csv, args.face_root, args.frame_root, "train")
    val_entries = _collect_or_load_entries(args.val_csv, args.face_root, args.frame_root, "val")
    test_entries = []
    if args.enable_solo_eval:
        test_entries = _collect_or_load_entries(args.test_csv, args.face_root, args.frame_root, "test")
    dial_test_entries = _collect_or_load_entries(args.dial_test_csv, args.dial_face_root, args.dial_frame_root, "dial_test")
    # 仅训练/评估 A/N/J/S：在 baseline_data 中它们原编码为 0/1/2/3；D/F/W 在此被跳过
    keep_labels = [0, 1, 2, 3]
    label_name_map_all = {0: "A", 1: "N", 2: "J", 3: "S", 4: "D", 5: "F", 6: "W"}
    train_entries_raw_n = len(train_entries)
    val_entries_raw_n = len(val_entries)
    test_raw_n = len(test_entries)
    dial_test_raw_n = len(dial_test_entries)
    dial_dropped = describe_dropped_entries_by_label(dial_test_entries, keep_labels)
    train_entries = filter_entries_by_labels(train_entries, keep_labels)
    val_entries = filter_entries_by_labels(val_entries, keep_labels)
    if args.enable_solo_eval:
        test_entries = filter_entries_by_labels(test_entries, keep_labels)
    dial_test_entries = filter_entries_by_labels(dial_test_entries, keep_labels)
    if not train_entries:
        diagnose_eanwh_no_entries(args.train_csv, args.face_root, args.frame_root)
        raise SystemExit("train_csv 无有效 EANwH 样本（需 face_root 与 frame_root 均有对应帧）。")
    if args.enable_solo_eval:
        print(
            f"Train sequences: {len(train_entries)} / {train_entries_raw_n}  "
            f"Val: {len(val_entries)} / {val_entries_raw_n}  "
            f"Solo test clips: {len(test_entries)} / {test_raw_n}"
        )
    else:
        print(
            f"Train sequences: {len(train_entries)} / {train_entries_raw_n}  "
            f"Val: {len(val_entries)} / {val_entries_raw_n}  "
            f"Solo test: disabled"
        )
    print(
        f"Dial test clips(A/N/J/S only): {len(dial_test_entries)} / "
        f"{dial_test_raw_n}"
    )
    if dial_dropped:
        print(f"[Dial drop] filtered by label: {len(dial_dropped)}")
        for i, d in enumerate(dial_dropped[:10], start=1):
            lname = label_name_map_all.get(d["label"], str(d["label"]))
            print(
                f"  {i}. label={lname}({d['label']}) stem={d['stem']} "
                f"video_key={d['video_key']} clip_hint={d['clip_hint']}"
            )
        if len(dial_dropped) > 10:
            print(f"  ... and {len(dial_dropped) - 10} more")
    print("Skip classes in this run: D/F/W")

    hand_feat_dir = args.hand_feat_dir or None
    train_ds = SequenceFrameDatasetEANwH(train_entries, img_size=args.img_size, hand_feat_dir=hand_feat_dir)
    val_ds = SequenceFrameDatasetEANwH(val_entries, img_size=args.img_size, hand_feat_dir=hand_feat_dir)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = None
    if args.enable_solo_eval:
        test_ds = SequenceFrameDatasetEANwH(test_entries, img_size=args.img_size, hand_feat_dir=hand_feat_dir)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    dial_hand_feat_dir = (args.dial_hand_feat_dir or "").strip() or None
    dial_test_ds = SequenceFrameDatasetEANwH(
        dial_test_entries, img_size=args.img_size, hand_feat_dir=dial_hand_feat_dir
    )
    dial_test_loader = DataLoader(
        dial_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = EANwHClassifier(
        num_classes=len(LABEL_NAMES),
        backbone_pretrain=args.backbone_pretrain,
        vggface2_ckpt=args.vggface2_ckpt or None,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    topk_records = []
    start_epoch = 0
    resume_path = Path(args.resume_path) if args.resume_path else None
    if resume_path is not None and resume_path.is_file():
        ckpt = torch.load(resume_path, map_location=device)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if isinstance(ckpt, dict):
            start_epoch = int(ckpt.get("epoch", 0))
            best_val_acc = float(ckpt.get("val_acc", best_val_acc))
        print(f"[Resume] loaded {resume_path}, start_epoch={start_epoch}, best_val_acc={best_val_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for faces, hands, labels in pbar:
            faces, hands = faces.to(device), hands.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(faces, hands)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)

            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device, return_f1=True)
        print(f"[Epoch {epoch + 1}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        print(f"[Dev set]  acc={val_acc:.4f}  f1={val_f1:.4f}")

        if args.enable_solo_eval and (epoch + 1) % max(args.solo_eval_every, 1) == 0:
            test_acc, test_f1, test_golds, test_preds = evaluate(model, test_loader, device, return_f1=True)
            print(f"[Solo test] acc={test_acc:.4f}  f1={test_f1:.4f}")
            print_and_save_classification_report(
                test_golds, test_preds,
                Path(args.save_dir), f"solo_epoch_{epoch + 1}",
                epoch + 1, test_acc, test_f1,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        topk_records = save_topk_by_val_acc(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            val_acc=val_acc,
            args=args,
            save_dir=Path(args.save_dir),
            topk_records=topk_records,
            k=5,
        )

        if (epoch + 1) % max(args.dial_eval_every, 1) == 0:
            dial_acc, dial_f1, dial_golds, dial_preds = evaluate(model, dial_test_loader, device, return_f1=True)
            print(f"[Dial test] acc={dial_acc:.4f}  f1={dial_f1:.4f}")
            print_and_save_classification_report(
                dial_golds, dial_preds,
                Path(args.save_dir), f"dial_epoch_{epoch + 1}",
                epoch + 1, dial_acc, dial_f1,
            )

    print(f"Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
