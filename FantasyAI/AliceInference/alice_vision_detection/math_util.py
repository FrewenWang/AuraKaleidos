def compute_iou(bbox, id1, id_keep):
    """
    :param bbox: 检测框的人脸框
    :param id1:
    :param id_keep:
    :return:
    """
    box1 = bbox[id1][:-1]  # * 5:id1 * 5 + 4]
    box2 = bbox[id_keep][:-1]  # [id_keep*5:id_keep*5+4]
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    l = max(l1, l2)
    t = max(t1, t2)
    r = min(r1, r2)
    b = min(b1, b2)

    if r <= l or b <= t:
        return 0.
    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)
    area_inter = (r - l) * (b - t)
    area_iou = area_inter / (area1 + area2 - area_inter)
    return area_iou
