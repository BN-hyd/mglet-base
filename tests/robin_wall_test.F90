PROGRAM robin_wall_test
    USE precision_mod, ONLY: realk
    USE robin_wall_mod, ONLY: robin_wall
    IMPLICIT NONE(type, external)
    REAL(realk) :: cw, flux, conductance, d, ks, eq, cp, ell, tol, exact
    INTEGER :: n

    tol = 100.0_realk*EPSILON(tol)
    eq = 0.2_realk
    cp = 1.3_realk
    ell = 0.7_realk
    DO n = 1, 3
        d = 0.1_realk*REAL(n, realk)
        ks = 0.3_realk/REAL(n, realk)
        CALL robin_wall(cw, flux, conductance, cp, ell, d, ks, eq)
        exact = (cp-eq)/(ell/d + 1.0_realk/ks)
        CALL check(ABS(flux-exact) < tol, "planar resistance")
        CALL check(ABS(flux-ks*(cw-eq)) < tol, "reaction law")
        CALL check(ABS(flux-d*(cp-cw)/ell) < tol, "diffusive law")
        CALL robin_wall(cw, flux, conductance, eq, ell, d, ks, eq)
        CALL check(ABS(flux) < tol, "equilibrium")
        CALL robin_wall(cw, flux, conductance, cp, ell, d, &
            0.0_realk, eq)
        CALL check(flux == 0.0_realk, "zero rate")
        CALL check(cw == cp, "zero gradient")
        CALL robin_wall(cw, flux, conductance, cp, ell, d, &
            SQRT(HUGE(ks)), eq)
        CALL check(ABS(cw-eq) < tol, "Dirichlet limit")
        CALL check(ABS(flux-d*(cp-eq)/ell) < tol, "strong flux")
        CALL robin_wall(cw, flux, conductance, -cp, ell, d, ks, eq)
        CALL check(flux < 0.0_realk, "reverse flux")
    END DO
    PRINT *, "Robin wall kernel: PASS"
CONTAINS
    SUBROUTINE check(ok, name)
        LOGICAL, INTENT(in) :: ok
        CHARACTER(len=*), INTENT(in) :: name
        IF (.NOT. ok) THEN
            PRINT *, "FAIL: ", name
            ERROR STOP 1
        END IF
    END SUBROUTINE check
END PROGRAM robin_wall_test
