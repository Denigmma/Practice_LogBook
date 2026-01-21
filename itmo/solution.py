"""15 task."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ORDERS_FILE = Path("orders.txt")
VALID_OUT_FILE = Path("order_country.txt")
INVALID_OUT_FILE = Path("non_valid_orders.txt")

PHONE_RE = re.compile(r"^\+\d-\d{3}-\d{3}-\d{2}-\d{2}$")
PRIORITY_ORDER = {"MAX": 0, "MIDDLE": 1, "LOW": 2}

ADDRESS_PARTS_COUNT = 4
ORDER_COLUMNS_COUNT = 6
RUS_COUNTRIES = {"Российская Федерация", "Россия"}


@dataclass(frozen=True)
class Order:
    """Order with validated and parsed fields."""

    order_id: str
    products_raw: str
    customer_name: str
    country: str
    region: str
    city: str
    street: str
    phone: str
    priority: str

    @property
    def address_short(self) -> str:
        """Return address in format: Region. City. Street."""
        return f"{self.region}. {self.city}. {self.street}"


def parse_products(products_raw: str) -> str:
    """Format products list with counts, preserving first appearance order."""
    items = [p.strip() for p in products_raw.split(",") if p.strip()]
    counts: dict[str, int] = {}
    seen_order: list[str] = []

    for item in items:
        if item not in counts:
            seen_order.append(item)
            counts[item] = 0
        counts[item] += 1

    parts: list[str] = []
    for item in seen_order:
        cnt = counts[item]
        parts.append(f"{item} x{cnt}" if cnt > 1 else item)

    return ", ".join(parts)


def parse_address(address: str) -> tuple[str, str, str, str] | None:
    """Parse address 'Country. Region. City. Street'.

    Return None if invalid.
    """
    if not address.strip():
        return None

    parts = [p.strip() for p in address.split(".")]
    parts = [p for p in parts if p]

    if len(parts) != ADDRESS_PARTS_COUNT:
        return None

    country, region, city, street = parts
    if not all((country, region, city, street)):
        return None

    return country, region, city, street


def validate_phone(phone: str) -> bool:
    """Return True if phone matches +x-xxx-xxx-xx-xx template."""
    phone_stripped = phone.strip()
    return bool(phone_stripped) and bool(PHONE_RE.match(phone_stripped))


def split_order_line(line: str) -> list[str] | None:
    """Split line by ';' into 6 columns or return None if malformed."""
    cols = [c.strip() for c in line.strip().split(";")]
    if len(cols) != ORDER_COLUMNS_COUNT:
        return None
    return cols


def country_sort_key(country: str) -> tuple[int, str]:
    """Sort key: Russia first, then alphabetical by country name."""
    normalized = country.strip()
    if normalized in RUS_COUNTRIES:
        return 0, ""
    return 1, normalized.lower()


def read_orders(path: Path) -> list[str]:
    """Read orders file lines, supporting utf-8 and cp1251 encodings."""
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp1251")

    return [line for line in text.splitlines() if line.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    """Write lines to a file, ending with a newline if there is any content."""
    content = "\n".join(lines)
    if lines:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def process_orders(
    lines: list[str],
) -> tuple[list[Order], list[tuple[str, int, str]]]:
    """Validate orders and return valid orders and validation error records."""
    valid_orders: list[Order] = []
    invalid_records: list[tuple[str, int, str]] = []

    for raw_line in lines:
        cols = split_order_line(raw_line)
        if cols is None:
            continue

        order_id, products, fio, address, phone, priority = cols

        address_parsed = parse_address(address)
        phone_ok = validate_phone(phone)

        if not phone_ok:
            bad_value = "no data" if not phone.strip() else phone
            invalid_records.append((order_id, 2, bad_value))

        if address_parsed is None:
            bad_value = "no data" if not address.strip() else address
            invalid_records.append((order_id, 1, bad_value))

        if address_parsed is None or not phone_ok:
            continue

        if priority not in PRIORITY_ORDER:
            continue

        country, region, city, street = address_parsed
        valid_orders.append(
            Order(
                order_id=order_id,
                products_raw=products,
                customer_name=fio,
                country=country,
                region=region,
                city=city,
                street=street,
                phone=phone,
                priority=priority,
            ),
        )

    return valid_orders, invalid_records


def main() -> None:
    """Run reading, validation, sorting, and writing of output files."""
    lines = read_orders(ORDERS_FILE)
    valid_orders, invalid_records = process_orders(lines)

    invalid_lines = [
        f"{order_id};{err_type};{bad}"
        for order_id, err_type, bad in invalid_records
    ]
    write_lines(INVALID_OUT_FILE, invalid_lines)

    valid_orders_sorted = sorted(
        valid_orders,
        key=lambda order: (
            country_sort_key(order.country),
            PRIORITY_ORDER[order.priority],
            order.order_id,
        ),
    )

    out_lines: list[str] = []
    for order in valid_orders_sorted:
        products_fmt = parse_products(order.products_raw)
        out_lines.append(
            f"{order.order_id};{products_fmt};{order.customer_name};"
            f"{order.address_short};{order.phone};{order.priority}",
        )

    write_lines(VALID_OUT_FILE, out_lines)


if __name__ == "__main__":
    main()
