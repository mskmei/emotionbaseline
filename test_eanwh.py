"""
EANwH 测试：在 test.csv 上评估，同一视频多段 logits 取平均再 argmax。
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from functions.baseline_data import (
    SequenceFrameDatasetEANwH,
    collect_sequence_entries_eanwh,
    filter_csv_with_face_and_hand,
)
from functions.baseline_torch_model import EANwHClassifier

LABEL_NAMES = ["A", "N", "J", "S"]


def save_eval_reports(all_golds, all_preds, save_dir: Path, prefix: str):
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
    np.save(save_dir / f"{prefix}_confusion_matrix.npy", cm)
    with open(save_dir / f"{prefix}_confusion_matrix.txt", "w", encoding="utf-8") as f:
        f.write("labels: " + ",".join(LABEL_NAMES) + "\n")
        for row in cm:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
    with open(save_dir / f"{prefix}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[Test] confusion matrix labels: {LABEL_NAMES}")
    print(cm)
    print("[Test] classification report:")
    print(report)
    print(f"[Test] classification report saved to: {save_dir / f'{prefix}_classification_report.txt'}")


def main():
    p = argparse.ArgumentParser(description="EANwH 测试")
    p.add_argument("--face_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl/face")
    p.add_argument("--frame_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl/frame")
    p.add_argument("--hand_feat_dir", type=str, default="",
                   help="预抽取手部特征目录（与训练一致）")
    p.add_argument("--hand_check_root", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl/hand",
                   help="用于过滤 CSV 的 hand npy 目录（<stem>.npy）")
    p.add_argument("--test_csv", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl/test_clips_balanced.csv")
    p.add_argument("--updated_csv_out", type=str, default="/raid_zoe/home/lr/wangyi/sign/bobsl/test_clips_balanced_updated.csv")
    p.add_argument("--path_ckpt", type=str, default="outputs_eanwh/best.pt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--win", type=int, default=10)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--hand_stride", type=int, default=1)
    p.add_argument("--save_dir", type=str, default="outputs_eanwh")
    p.add_argument("--report_prefix", type=str, default="dial_eval")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.updated_csv_out:
        args.test_csv = filter_csv_with_face_and_hand(
            csv_path=args.test_csv,
            face_root=args.face_root,
            hand_root=args.hand_check_root or args.hand_feat_dir,
            frame_root=args.frame_root,
            hand_stride=args.hand_stride,
            output_csv=args.updated_csv_out,
        )
    entries = collect_sequence_entries_eanwh(
        args.test_csv, args.face_root, args.frame_root, win=args.win, step=args.step
    )
    # 在 collect_sequence_entries_eanwh 之后，添加：
    keep_labels = [0, 1, 2, 3]

    def filter_entries_by_labels(entries, keep_labels):
        old_to_new = {old: new for new, old in enumerate(keep_labels)}
        out = []
        for e in entries:
            if e["label"] not in old_to_new:
                continue
            ne = dict(e)
            ne["label"] = old_to_new[ne["label"]]
            out.append(ne)
        return out

    entries = filter_entries_by_labels(entries, keep_labels)
    if not entries:
        print("test_csv 无有效 EANwH 样本")
        return

    by_video = defaultdict(list)
    for e in entries:
        by_video[e["stem"]].append(e)

    model = EANwHClassifier(num_classes=4, backbone_pretrain="none").to(device)
    ckpt = torch.load(args.path_ckpt, map_location=device)
    if isinstance(ckpt, dict):
        print(
            f"[CKPT] path={args.path_ckpt} "
            f"epoch={ckpt.get('epoch', 'NA')} "
            f"val_acc={ckpt.get('val_acc', 'NA')}"
        )
        if "args" in ckpt and isinstance(ckpt["args"], dict):
            ckpt_args = ckpt["args"]
            print(
                f"[CKPT] train_csv={ckpt_args.get('train_csv', 'NA')} "
                f"val_csv={ckpt_args.get('val_csv', 'NA')} "
                f"dial_test_csv={ckpt_args.get('dial_test_csv', 'NA')}"
            )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    hand_feat_dir = args.hand_feat_dir or None
    correct_videos, total_videos = 0, 0
    all_preds, all_golds = [], []
    with torch.no_grad():
        all_logits_list = []
        for stem, stem_entries in by_video.items():
            label = stem_entries[0]["label"]
            ds = SequenceFrameDatasetEANwH(stem_entries, img_size=args.img_size, hand_feat_dir=hand_feat_dir)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            all_logits = []
            for faces, hands, _ in loader:
                faces, hands = faces.to(device), hands.to(device)
                logits = model(faces, hands)
                all_logits.append(logits)
            if not all_logits:
                continue
            logits_cat = torch.cat(all_logits, dim=0)
            logits_mean = logits_cat.mean(dim=0, keepdim=True)
            pred = logits_mean.argmax(1).item()
            correct_videos += 1 if pred == label else 0
            total_videos += 1
            all_preds.append(pred)
            all_golds.append(label)
            logits_mean = logits_cat.mean(dim=0, keepdim=True)
            all_logits_list.append(logits_mean.cpu().numpy())
        all_logits_np = np.concatenate(all_logits_list, axis=0)  # (N, 4)
        print("[Logits mean per class]", all_logits_np.mean(axis=0))
        print("[Logits std  per class]", all_logits_np.std(axis=0))
        print("[Logits min  per class]", all_logits_np.min(axis=0))
        print("[Logits max  per class]", all_logits_np.max(axis=0))


    acc = correct_videos / max(total_videos, 1)
    print(f"[Test] {args.test_csv}  videos={total_videos}  acc={acc:.4f} ({correct_videos}/{total_videos})  [策略: 平均 logits 再 argmax]")
    labels_idx = list(range(len(LABEL_NAMES)))
    support = metrics.confusion_matrix(all_golds, all_preds, labels=labels_idx).sum(axis=1).tolist()
    print(f"[Test] support(A,N,J,S)={support} total={sum(support)}")
    gold_counts = np.bincount(np.array(all_golds, dtype=np.int64), minlength=len(LABEL_NAMES)).tolist()
    pred_counts = np.bincount(np.array(all_preds, dtype=np.int64), minlength=len(LABEL_NAMES)).tolist()
    print(f"[Test] gold_counts(A,N,J,S)={gold_counts}")
    print(f"[Test] pred_counts(A,N,J,S)={pred_counts}")
    save_eval_reports(all_golds, all_preds, Path(args.save_dir), args.report_prefix)


if __name__ == "__main__":
    main()
