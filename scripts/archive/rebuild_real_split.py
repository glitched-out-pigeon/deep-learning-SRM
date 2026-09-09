import json
import random
from pathlib import Path
import zipfile


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "sen2naip"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

GROUP_FIELDS = [
    "s2_id",
    "naip_id",
    "roi_id",
    "proj:geometry",
]


# ============================================================
# FIND ZIP
# ============================================================

zip_files = list(
    RAW_ROOT.rglob("*.zip")
)

if not zip_files:
    raise FileNotFoundError(
        f"No SEN2NAIP ZIP found under:\n{RAW_ROOT}"
    )

ZIP_PATH = zip_files[0]


# ============================================================
# UNION-FIND
# ============================================================

class UnionFind:

    def __init__(self):

        self.parent = {}
        self.rank = {}

    def add(self, x):

        if x not in self.parent:

            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):

        if self.parent[x] != x:

            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("BUILDING LEAK-FREE SEN2NAIP SPLIT")
print("=" * 70)

print("\nZIP:")
print(ZIP_PATH)


with zipfile.ZipFile(
    ZIP_PATH,
    "r",
) as z:

    names = z.namelist()
    name_set = set(names)

    metadata = {}

    for name in names:

        if not name.startswith(
            "cross-sensor/"
        ):
            continue

        if not name.endswith(
            "/metadata.json"
        ):
            continue

        roi = name.split("/")[-2]

        lr_path = (
            f"cross-sensor/{roi}/lr.tif"
        )

        hr_path = (
            f"cross-sensor/{roi}/hr.tif"
        )

        if (
            lr_path not in name_set
            or hr_path not in name_set
        ):
            continue

        try:

            info = json.loads(
                z.read(name).decode("utf-8")
            )

            metadata[roi] = info

        except Exception as exc:

            print(
                f"WARNING: failed to read "
                f"{roi}: {exc}"
            )


print(
    "\nValid ROIs:",
    len(metadata)
)


# ============================================================
# BUILD CONNECTED COMPONENTS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING SOURCE CONNECTIVITY")
print("=" * 70)


uf = UnionFind()

# Every ROI gets a unique node.
for roi in metadata:

    roi_node = f"ROI::{roi}"

    uf.add(roi_node)


# Connect every ROI to all source identifiers.
#
# If two ROIs share ANY of these identifiers, they become
# part of the same connected component.

for roi, info in metadata.items():

    roi_node = f"ROI::{roi}"

    for field in GROUP_FIELDS:

        value = info.get(field)

        if value is None:
            continue

        node = (
            f"{field}::{value}"
        )

        uf.union(
            roi_node,
            node
        )


# ============================================================
# EXTRACT COMPONENTS
# ============================================================

components = {}

for roi in metadata:

    root = uf.find(
        f"ROI::{roi}"
    )

    components.setdefault(
        root,
        []
    ).append(roi)


components = list(
    components.values()
)

print(
    "\nConnected components:",
    len(components)
)


component_sizes = sorted(
    [len(c) for c in components],
    reverse=True,
)


print(
    "Largest component:",
    component_sizes[0]
)

print(
    "Smallest component:",
    component_sizes[-1]
)

print(
    "Average component size:",
    f"{sum(component_sizes) / len(component_sizes):.2f}"
)


# ============================================================
# SHUFFLE COMPONENTS
# ============================================================

random.seed(SEED)

random.shuffle(
    components
)


# ============================================================
# SPLIT COMPONENTS
# ============================================================

total_rois = len(metadata)

target_train = (
    total_rois * TRAIN_RATIO
)

target_val = (
    total_rois * VAL_RATIO
)

train_components = []
val_components = []
test_components = []

train_count = 0
val_count = 0


for component in components:

    size = len(component)

    # Fill train until approximately 80%.
    if train_count < target_train:

        train_components.append(
            component
        )

        train_count += size

    # Then validation.
    elif val_count < target_val:

        val_components.append(
            component
        )

        val_count += size

    # Remaining components → test.
    else:

        test_components.append(
            component
        )


# ============================================================
# BUILD ROI LISTS
# ============================================================

train_rois = sorted(
    roi
    for component in train_components
    for roi in component
)

val_rois = sorted(
    roi
    for component in val_components
    for roi in component
)

test_rois = sorted(
    roi
    for component in test_components
    for roi in component
)


splits = {

    "train": train_rois,
    "val": val_rois,
    "test": test_rois,

}


# ============================================================
# PRINT SPLIT
# ============================================================

print("\n" + "=" * 70)
print("NEW CONNECTED-COMPONENT SPLIT")
print("=" * 70)

for split, rois in splits.items():

    print(
        f"\n{split.upper()}:"
    )

    print(
        "  ROIs:",
        len(rois)
    )

    print(
        "  Components:",
        len({
            uf.find(f"ROI::{roi}")
            for roi in rois
        })
    )


# ============================================================
# GENERIC CROSS-SPLIT CHECK
# ============================================================

def check_field_disjointness(
    field,
):

    value_to_splits = {}

    for split, rois in splits.items():

        for roi in rois:

            value = metadata[roi].get(
                field
            )

            if value is None:
                continue

            value_to_splits.setdefault(
                value,
                set()
            ).add(
                split
            )

    conflicts = {

        value: split_set

        for value, split_set
        in value_to_splits.items()

        if len(split_set) > 1

    }

    return conflicts


# ============================================================
# CHECK ALL LEAKAGE TYPES
# ============================================================

print("\n" + "=" * 70)
print("FINAL LEAKAGE CHECK")
print("=" * 70)


all_clear = True


for field in GROUP_FIELDS:

    conflicts = (
        check_field_disjointness(
            field
        )
    )

    print(
        f"\n{field}:"
    )

    print(
        "  Cross-split conflicts:",
        len(conflicts)
    )

    if conflicts:

        all_clear = False

        for value, split_set in list(
            conflicts.items()
        )[:5]:

            print(
                " ",
                split_set,
                str(value)[:180]
            )

    else:

        print(
            "  ✅ NONE"
        )


# ============================================================
# ROI CHECK
# ============================================================

train_set = set(train_rois)
val_set = set(val_rois)
test_set = set(test_rois)


roi_conflicts = (
    train_set & val_set
) | (
    train_set & test_set
) | (
    val_set & test_set
)


print(
    "\nROI:"
)

print(
    "  Cross-split conflicts:",
    len(roi_conflicts)
)


if roi_conflicts:

    all_clear = False

else:

    print(
        "  ✅ NONE"
    )


# ============================================================
# FINAL DECISION
# ============================================================

if not all_clear:

    print(
        "\n❌ LEAKAGE IS STILL PRESENT."
    )

    raise RuntimeError(
        "Leakage remains in the new split."
    )


print(
    "\n✅ ALL SPLIT LEAKAGE CHECKS PASSED."
)


# ============================================================
# SOURCE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SOURCE STATISTICS")
print("=" * 70)


for split, rois in splits.items():

    print(
        f"\n{split.upper()}:"
    )

    for field in GROUP_FIELDS:

        values = {
            metadata[roi].get(field)
            for roi in rois
            if metadata[roi].get(field)
            is not None
        }

        print(
            f"  {field}:",
            len(values)
        )


# ============================================================
# SAVE SPLIT DEFINITION
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SPLIT_FILE = (
    OUTPUT_ROOT
    / "leak_free_split.json"
)


split_definition = {

    "seed": SEED,

    "method": (
        "Connected-component grouping using "
        "S2 ID, NAIP ID, dataset ROI ID and "
        "exact geometry."
    ),

    "group_fields": GROUP_FIELDS,

    "train_rois": train_rois,

    "val_rois": val_rois,

    "test_rois": test_rois,

    "train_components": [
        metadata[roi]["s2_id"]
        for component in train_components
        for roi in component
        if "s2_id" in metadata[roi]
    ],

    "val_components": [
        metadata[roi]["s2_id"]
        for component in val_components
        for roi in component
        if "s2_id" in metadata[roi]
    ],

    "test_components": [
        metadata[roi]["s2_id"]
        for component in test_components
        for roi in component
        if "s2_id" in metadata[roi]
    ],
}


with open(
    SPLIT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        split_definition,
        f,
        indent=2,
    )


print("\n" + "=" * 70)
print("LEAK-FREE SPLIT SAVED")
print("=" * 70)

print(
    "\nPath:",
    SPLIT_FILE
)

print(
    "\n✅ This split is ready for dataset preparation."
)