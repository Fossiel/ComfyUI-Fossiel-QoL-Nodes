#!/usr/bin/env python3
"""
Standalone script extracting the exact aspect-ratio + resolution logic
from FossielResolutionWrangler (Automatic mode, Max Resolution, tolerance 8,
default max 1024×1024).

Only standard-library modules are used.
"""

import math
from fractions import Fraction


def find_closest_ratio(width, height, max_side=24):
    if width == 0 or height == 0:
        return 1, 1
    ratio = width / height
    landscape = width > height
    best_error = float('inf')
    best_loss = float('inf')
    best_a, best_b = 1, 1

    def continued_fraction(r):
        a = int(r)
        yield a
        frac = r - a
        seen = set()
        while frac > 1e-9 and len(seen) < 30:
            if frac in seen:
                break
            seen.add(frac)
            r = 1 / frac
            a = int(r)
            yield a
            frac = r - a

    cf = list(continued_fraction(ratio))
    convergents = []
    p0, q0, p1, q1 = 0, 1, 1, 0
    for a in cf:
        p = a * p1 + p0
        q = a * q1 + q0
        convergents.append((p, q))
        p0, q0, p1, q1 = p1, q1, p, q

    for a, b in convergents:
        if max(a, b) > max_side:
            continue
        if landscape and a <= b:
            continue
        if not landscape and b <= a:
            continue
        g = math.gcd(a, b)
        a_r, b_r = a // g, b // g
        if a_r == 0 or b_r == 0:
            continue
        error = abs(Fraction(a_r, b_r) - Fraction(width, height))
        if landscape:
            loss = abs(width - height * Fraction(a_r, b_r))
        else:
            loss = abs(height - width * Fraction(b_r, a_r))
        if error < best_error or (abs(error - best_error) < 1e-10 and loss < best_loss):
            best_error = error
            best_loss = loss
            best_a, best_b = a_r, b_r

    if best_a == 1 and best_b == 1:
        if landscape:
            best_a, best_b = 4, 3
        else:
            if height > width * 2.5:
                best_a, best_b = 2, 3
            elif height > width * 1.5:
                best_a, best_b = 3, 4
            else:
                best_a, best_b = 3, 4
    return best_a, best_b


def expand_and_crop_size(orig_w, orig_h, target_num, target_den, aspect_method="Automatic"):
    """Return the cropped (aspect-matched) width and height – pure size logic."""
    if aspect_method == "Automatic" and orig_w == orig_h:
        return orig_w, orig_h

    orig_ratio = orig_w / orig_h
    targ_ratio = target_num / target_den
    if orig_ratio > targ_ratio:
        fit_units = orig_h // target_den
    else:
        fit_units = orig_w // target_num
    new_w = fit_units * target_num
    new_h = fit_units * target_den
    return new_w, new_h


def resize_to_divisible(base_w, base_h, aspect_num, aspect_den, tolerance, pixel_cap):
    # === SPECIAL CASE FOR 1:1 (Perfect Square) ===
    if aspect_num == 1 and aspect_den == 1:
        max_side = int(math.sqrt(pixel_cap))
        for side in range(max_side, 0, -1):
            if side % tolerance == 0:
                if side * side <= pixel_cap:
                    return side, side
        return tolerance, tolerance

    # === ORIGINAL CODE FOR NON-1:1 RATIOS ===
    w = base_w
    h = base_h
    max_w, max_h = w, h
    max_pixels = w * h

    while True:
        next_w = w + aspect_num
        next_h = h + aspect_den
        if next_w * next_h > pixel_cap:
            break
        w, h = next_w, next_h
        if w % tolerance == 0 and h % tolerance == 0:
            current_pixels = w * h
            if current_pixels > max_pixels:
                max_w, max_h = w, h
                max_pixels = current_pixels

    return max_w, max_h


def calculate(width, height,
              max_resolution_x=1024,
              max_resolution_y=1024,
              aspect_tolerance=8):
    """
    Exact Automatic / Max-Resolution path from the node.
    Returns:
        out_aspect_x, out_aspect_y,
        aspect_w, aspect_h,          # size after aspect crop
        final_w, final_h             # final resolution under the pixel cap
    """

    tolerance = int(aspect_tolerance)

    # 1. Closest integer aspect ratio
    target_num, target_den = find_closest_ratio(width, height, max_side=24)

    # 2. Cropped (aspect-matched) size
    aspect_w, aspect_h = expand_and_crop_size(
        width, height, target_num, target_den, aspect_method="Automatic"
    )

    # Simplify aspect after crop (exactly as the node does)
    if aspect_w != 0 and aspect_h != 0:
        gcd_crop = math.gcd(aspect_w, aspect_h)
        target_num = aspect_w // gcd_crop
        target_den = aspect_h // gcd_crop

    cropped_pixels = aspect_w * aspect_h
    effective_pixel_cap = max_resolution_x * max_resolution_y

    # 3. Minimal divisible size
    min_w = target_num * tolerance
    min_h = target_den * tolerance

    base_w = aspect_w
    base_h = aspect_h

    if cropped_pixels > effective_pixel_cap or not (
        aspect_w % tolerance == 0 and aspect_h % tolerance == 0
    ):
        max_units = min(
            base_w // target_num,
            base_h // target_den,
            int((effective_pixel_cap ** 0.5) / max(target_num, target_den, 1))
        )
        while max_units > 0:
            test_w = max_units * target_num
            test_h = max_units * target_den
            if test_w * test_h <= effective_pixel_cap:
                base_w = test_w
                base_h = test_h
                break
            max_units -= 1
        if base_w < min_w or base_h < min_h:
            base_w = min_w
            base_h = min_h

    base_w = max(base_w, min_w)
    base_h = max(base_h, min_h)

    # 4. Final size that is divisible by tolerance and respects the pixel cap
    final_w, final_h = resize_to_divisible(
        base_w, base_h, target_num, target_den, tolerance, effective_pixel_cap
    )

    # 5. Simplified aspect for output
    gcd = math.gcd(target_num, target_den)
    out_aspect_x = target_num // gcd
    out_aspect_y = target_den // gcd

    return (
        out_aspect_x, out_aspect_y,
        aspect_w, aspect_h,
        final_w, final_h
    )


def fit_size_to_aspect():
    try:
        print()
        print("\n=== Calculate Closest Size Given Coherent Aspect and 1 Side ===")
        print()
        aspect_ratio = input("Enter target aspect ratio (e.g., 16:9, 4:3): ").strip().split(':')
        if len(aspect_ratio) != 2:
            print("Invalid aspect ratio format.")
            return

        target_num = int(aspect_ratio[0])
        target_den = int(aspect_ratio[1])

        choose_side = input("Choose which side to start with (w/h): ").strip().lower()
        if choose_side not in ['w', 'h']:
            print("Invalid choice. Please enter 'w' or 'h'.")
            return

        chosen_side_length = int(input("Enter the length of the selected side: "))

        # Calculate closest compatible side length
        if choose_side == 'w':
            calculated_side = int((chosen_side_length / target_num) + 0.5) * target_num
            perpendicular_side = calculated_side / target_num * target_den
        else:
            calculated_side = int((chosen_side_length / target_den) + 0.5) * target_den
            perpendicular_side = calculated_side / target_den * target_num

        # Output results
        print(f"\nResult:")
        if choose_side == 'w':
            print(f"Width: {calculated_side}")
            print(f"Height: {int(perpendicular_side)}")
        else:
            print(f"Width: {int(perpendicular_side)}")
            print(f"Height: {calculated_side}")
        print(f"Aspect Ratio: {target_num}:{target_den}")

    except ValueError:
        print("Please enter valid integers.")
        return


if __name__ == "__main__":
    print("Fossiel Resolution Wrangler – standalone aspect & resolution calculator")
    print()

    while True:
        print("\n--- Main Menu ---")
        print("1. Best Fitting Coherent Aspect Ratio for Given Size")
        print("2. Closest Size Given Coherent Aspect and 1 Side")
        print("3. Exit")

        choice = input("\nEnter your choice (1/2/3): ").strip()

        if choice == '1':
            try:
                print()
                print("\n=== Calculate Best Fitting Coherent Aspect Ratio for Given Size ===")
                print()

                width = int(input("Enter width: ").strip())
                height = int(input("Enter height: ").strip())
            except ValueError:
                print("Please enter valid integers.")
                continue

            if width <= 0 or height <= 0:
                print("Width and height must be positive.")
                continue

            aspect_x, aspect_y, crop_w, crop_h, final_w, final_h = calculate(width, height)

            print()
            print(f"Input size          : {width} × {height}")
            print(f"Best integer aspect : {aspect_x}:{aspect_y}")
            print(f"Aspect resolution   : {crop_w} × {crop_h}")
            print(f"Pixel count         : {crop_w * crop_h}")

        elif choice == '2':
            fit_size_to_aspect()

        elif choice == '3':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
