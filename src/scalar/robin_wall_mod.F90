MODULE robin_wall_mod
    USE precision_mod, ONLY: realk

    IMPLICIT NONE(type, external)
    PRIVATE
    PUBLIC :: robin_wall

CONTAINS
    PURE SUBROUTINE robin_wall(cwall, jout, conductance, cbar, ell, &
            diffusivity, k_s, c_eq)
        ! For the existing normal least-squares reconstruction:
        ! ell = SUM(d**2)/SUM(d), cbar = SUM(d*C)/SUM(d), d > 0.
        ! C(s) = Cw + s*g, g = (cbar-Cw)/ell, D*g = k_s*(Cw-Ceq).
        ! Use bounded ratios (no k_s*ell product for large k_s).
        REAL(realk), INTENT(out) :: cwall, jout, conductance
        REAL(realk), INTENT(in) :: cbar, ell, diffusivity, k_s, c_eq
        REAL(realk) :: gd, ratio, weight

        gd = diffusivity/ell
        IF (k_s == 0.0_realk) THEN
            conductance = 0.0_realk
            weight = 1.0_realk
        ELSE IF (k_s <= gd) THEN
            ratio = k_s/gd
            weight = 1.0_realk/(1.0_realk + ratio)
            conductance = k_s*weight
        ELSE
            ratio = gd/k_s
            weight = ratio/(1.0_realk + ratio)
            conductance = gd/(1.0_realk + ratio)
        END IF
        cwall = weight*cbar + (1.0_realk - weight)*c_eq
        jout = conductance*(cbar - c_eq)
    END SUBROUTINE robin_wall
END MODULE robin_wall_mod
