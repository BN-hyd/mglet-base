PROGRAM robin_wall_test
    USE precision_mod, ONLY: realk
    USE robin_wall_mod, ONLY: robin_wall
    IMPLICIT NONE(type, external)
    REAL(realk) :: cw, flux, conductance, d, ks, eq, cp, ell, tol, exact
    REAL(realk) :: distances(3), observations(3), weights(3), target
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
        CALL check(ABS(flux) <= 0.0_realk, "zero rate")
        CALL check(ABS(cw-cp) < tol, "zero gradient")
        CALL robin_wall(cw, flux, conductance, cp, ell, d, &
            SQRT(HUGE(ks)), eq)
        CALL check(ABS(cw-eq) < tol, "Dirichlet limit")
        CALL check(ABS(flux-d*(cp-eq)/ell) < tol, "strong flux")
        CALL robin_wall(cw, flux, conductance, -cp, ell, d, ks, eq)
        CALL check(flux < 0.0_realk, "reverse flux")
    END DO
    ! Nonuniform multipoint normal reconstruction of a known linear field.
    distances = [0.1_realk, 0.3_realk, 0.8_realk]
    weights = distances/SUM(distances)
    ell = SUM(distances**2)/SUM(distances)
    d = 0.4_realk
    ks = 1.7_realk
    target = 0.2_realk
    observations = eq + target/ks + target*distances/d
    cp = SUM(weights*observations)
    CALL robin_wall(cw, flux, conductance, cp, ell, d, ks, eq)
    CALL check(ABS(flux-target) < tol, "multipoint normal fit")
    CALL check(ABS(cw-eq-target/ks) < tol, "multipoint wall value")
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
