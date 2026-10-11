MODULE gc_robin_mod
    USE core_mod
    USE ib_mod, ONLY: gc_t, wmindexlistn, wmcheckneighbor, choosestencil
    USE scacore_mod, ONLY: scalar, scalar_t, nsca
    USE flow_mod, ONLY: gmol, rho
    USE robin_wall_mod, ONLY: robin_wall
    USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_is_finite

    IMPLICIT NONE(type, external)
    PRIVATE

    TYPE :: robin_patch_t
        INTEGER(intk) :: owner(3), body, n
        INTEGER(intk) :: point(3, 7)
        REAL(realk) :: weight(7), ell, area, volume
    END TYPE robin_patch_t

    TYPE :: robin_grid_t
        TYPE(robin_patch_t), ALLOCATABLE :: patch(:)
        REAL(realk) :: diffusion_rate
    END TYPE robin_grid_t

    TYPE(robin_grid_t), ALLOCATABLE :: grids(:)

    PUBLIC :: create_robin, finish_robin, apply_robin

CONTAINS
    SUBROUTINE create_robin(gc)
        TYPE(gc_t), INTENT(in) :: gc
        INTEGER(intk) :: m, igrid, kk, jj, ii, ip3, ipp, ncells
        INTEGER(intk) :: bconds(6)
        REAL(realk), POINTER, CONTIGUOUS :: bp(:, :, :), bt(:, :, :)
        REAL(realk), POINTER, CONTIGUOUS :: x(:), y(:), z(:)
        REAL(realk), POINTER, CONTIGUOUS :: dx(:), dy(:), dz(:)

        IF (.NOT. ANY(scalar%has_robin)) RETURN
        ALLOCATE(grids(nmygrids))
        DO m = 1, nmygrids
            igrid = mygrids(m)
            CALL get_mgdims(kk, jj, ii, igrid)
            CALL get_ip3(ip3, igrid)
            CALL get_mgbasb(bconds, igrid)
            CALL get_fieldptr(bp, "BP", igrid)
            CALL get_fieldptr(bt, "BT", igrid)
            CALL get_fieldptr(x, "X", igrid)
            CALL get_fieldptr(y, "Y", igrid)
            CALL get_fieldptr(z, "Z", igrid)
            CALL get_fieldptr(dx, "DDX", igrid)
            CALL get_fieldptr(dy, "DDY", igrid)
            CALL get_fieldptr(dz, "DDZ", igrid)
            ipp = gc%icellspointer(igrid)
            ncells = gc%icells(igrid)
            CALL create_grid(grids(m), kk, jj, ii, bp, bt, x, y, z, &
                dx, dy, dz, bconds, gc%bzelltyp(ip3:ip3+kk*jj*ii-1), &
                gc%nvecs(:, ipp:ipp+ncells-1), &
                gc%arealist(ipp:ipp+ncells-1), &
                gc%bodyid(ipp:ipp+ncells-1), &
                gc%ucell(:, ipp:ipp+ncells-1))
        END DO
    END SUBROUTINE create_robin


    SUBROUTINE create_grid(grid, kk, jj, ii, bp, bt, x, y, z, dx, dy, &
            dz, bconds, bzelltyp, nvecs, areas, bodies, velocity)
        TYPE(robin_grid_t), INTENT(out) :: grid
        INTEGER(intk), INTENT(in) :: kk, jj, ii, bconds(6)
        REAL(realk), INTENT(in) :: bp(kk, jj, ii), bt(kk, jj, ii)
        REAL(realk), INTENT(in) :: x(ii), y(jj), z(kk)
        REAL(realk), INTENT(in) :: dx(ii), dy(jj), dz(kk)
        INTEGER(intk), INTENT(in) :: bzelltyp(kk, jj, ii), bodies(:)
        REAL(realk), INTENT(in) :: nvecs(:, :), areas(:), velocity(:, :)
        TYPE(robin_patch_t), ALLOCATABLE :: work(:)
        INTEGER(intk) :: i, j, k, idx, body, p, l, n, count, chosen
        INTEGER(intk) :: ni(122), nj(122), nk(122), nn
        INTEGER(intk) :: si(122), sj(122), sk(122), nstc, fallback
        INTEGER(intk) :: ai(122), aj(122), ak(122), nearest
        REAL(realk) :: d, dc, ds(7), norm, radius, best
        LOGICAL :: reactive

        ALLOCATE(work(SIZE(areas)))
        CALL wmindexlistn(ni, nj, nk, nn)
        count = 0
        fallback = 0
        grid%diffusion_rate = 4.0_realk * &
            (1.0_realk/MINVAL(dx(3:ii-2))**2 &
            + 1.0_realk/MINVAL(dy(3:jj-2))**2 &
            + 1.0_realk/MINVAL(dz(3:kk-2))**2)
        DO i = 3, ii-2
            DO j = 3, jj-2
                DO k = 3, kk-2
                    IF (bzelltyp(k, j, i) >= 0) CYCLE
                    idx = -bzelltyp(k, j, i)
                    body = bodies(idx)
                    IF (body <= 0) THEN
                        WRITE(*, *) "Robin: unresolved surface body id"
                        CALL errr(__FILE__, __LINE__)
                    END IF
                    reactive = .FALSE.
                    DO l = 1, nsca
                        reactive = reactive .OR. &
                            (scalar(l)%geometries(body)%flag == 2 .AND. &
                            scalar(l)%geometries(body)%k_s > 0.0_realk)
                    END DO
                    IF (.NOT. reactive) CYCLE
                    IF (ANY(ABS(velocity(:, idx)) > 0.0_realk)) THEN
                        WRITE(*, *) "Robin supports stationary walls only"
                        CALL errr(__FILE__, __LINE__)
                    END IF
                    IF (ABS(areas(idx)) < TINY(1.0_realk)) CYCLE
                    norm = SQRT(SUM(nvecs(1:3, idx)**2))
                    IF (.NOT. ALL(ieee_is_finite(nvecs(:, idx))) .OR. &
                            .NOT. ieee_is_finite(norm) .OR. &
                            ABS(norm - 1.0_realk) > 1.0e-4_realk .OR. &
                            .NOT. ieee_is_finite(areas(idx)) .OR. &
                            areas(idx) < 0.0_realk .OR. &
                            bt(k, j, i) < 0.5_realk) THEN
                        WRITE(*, *) "Robin: invalid/unresolved patch", k, j, i
                        CALL errr(__FILE__, __LINE__)
                    END IF
                    count = count + 1
                    work(count)%owner = [k, j, i]
                    work(count)%body = body
                    work(count)%area = areas(idx)
                    work(count)%volume = dx(i)*dy(j)*dz(k)
                    work(count)%point = 1
                    work(count)%weight = 0.0_realk
                    ds = 0.0_realk
                    p = 0
                    dc = DOT_PRODUCT(nvecs(1:3, idx), [x(i), y(j), z(k)]) &
                        - nvecs(4, idx)
                    si = 0
                    sj = 0
                    sk = 0
                    CALL wmcheckneighbor(k, j, i, kk, jj, ii, bp, &
                        bconds, nn, ni, nj, nk, si, sj, sk, nstc)
                    ai = si
                    aj = sj
                    ak = sk
                    CALL choosestencil(k, j, i, nstc, si, sj, sk, chosen)
                    ! An open intersected cell is a valid observation too.
                    IF (bp(k, j, i) > 0.5_realk .AND. &
                            dc > TINY(dc)) THEN
                        p = 1
                        work(count)%point(:, p) = [k, j, i]
                        ds(p) = dc
                    END IF
                    DO n = 1, chosen
                        d = DOT_PRODUCT(nvecs(1:3, idx), &
                            [x(si(n)), y(sj(n)), z(sk(n))]) - nvecs(4, idx)
                        IF (d <= MAX(TINY(d), 0.2_realk*ABS(dc))) CYCLE
                        p = p + 1
                        work(count)%point(:, p) = [sk(n), sj(n), si(n)]
                        ds(p) = d
                    END DO
                    ! The legacy rescue silently imposes zero flux. Robin
                    ! instead uses one valid BP-open observation, or fails.
                    IF (p == 0) THEN
                        nearest = 0
                        best = HUGE(best)
                        DO n = 1, nstc
                            d = DOT_PRODUCT(nvecs(1:3, idx), &
                                [x(ai(n)), y(aj(n)), z(ak(n))]) &
                                - nvecs(4, idx)
                            IF (d <= MAX(TINY(d), &
                                    0.2_realk*ABS(dc))) CYCLE
                            radius = (x(ai(n))-x(i))**2 &
                                + (y(aj(n))-y(j))**2 + (z(ak(n))-z(k))**2
                            IF (radius < best) THEN
                                best = radius
                                nearest = n
                                ds(1) = d
                            END IF
                        END DO
                        IF (nearest == 0) THEN
                            WRITE(*, *) "Robin: no fluid stencil", k, j, i
                            CALL errr(__FILE__, __LINE__)
                        END IF
                        p = 1
                        n = nearest
                        work(count)%point(:, 1) = [ak(n), aj(n), ai(n)]
                        fallback = fallback + 1
                    END IF
                    work(count)%n = p
                    ! Normalize before squaring to avoid scale dependence.
                    d = MAXVAL(ds(1:p))
                    ds(1:p) = ds(1:p)/d
                    work(count)%ell = d*SUM(ds(1:p)**2)/SUM(ds(1:p))
                    work(count)%weight(1:p) = ds(1:p)/SUM(ds(1:p))
                END DO
            END DO
        END DO
        grid%patch = work(1:count)
        IF (ANY(scalar%robin_budget)) THEN
            DO body = 1, SIZE(scalar(1)%geometries)
                d = 0.0_realk
                p = 0
                DO n = 1, count
                    IF (work(n)%body /= body) CYCLE
                    d = d + work(n)%area
                    p = p + 1
                END DO
                WRITE(*, *) "ROBIN_PATCHES rank body count area", &
                    myid, body, p, d
            END DO
        END IF
        IF (fallback > 0) THEN
            WRITE(*, *) "Robin one-point fallback patches on rank", &
                myid, ":", fallback
        END IF
    END SUBROUTINE create_grid


    SUBROUTINE apply_robin(qtt, reaction, t, sca, dt, rkfactor)
        TYPE(field_t), INTENT(inout) :: qtt, reaction
        TYPE(field_t), INTENT(in) :: t
        TYPE(scalar_t), INTENT(in) :: sca
        REAL(realk), INTENT(in) :: dt, rkfactor
        REAL(realk), POINTER, CONTIGUOUS :: c(:, :, :), rhs(:, :, :)
        REAL(realk), POINTER, CONTIGUOUS :: rate(:, :, :)
        INTEGER(intk) :: m, igrid, p, n, k, j, i, body
        REAL(realk) :: cbar, cwall, jout, conductance, diffusivity, bound
        TYPE(robin_patch_t) :: patch

        reaction%arr = 0.0_realk
        diffusivity = gmol/rho/sca%prmol
        DO m = 1, nmygrids
            igrid = mygrids(m)
            CALL t%get_ptr(c, igrid)
            CALL qtt%get_ptr(rhs, igrid)
            CALL reaction%get_ptr(rate, igrid)
            DO p = 1, SIZE(grids(m)%patch)
                patch = grids(m)%patch(p)
                body = patch%body
                IF (sca%geometries(body)%flag /= 2) CYCLE
                cbar = 0.0_realk
                DO n = 1, patch%n
                    k = patch%point(1, n)
                    j = patch%point(2, n)
                    i = patch%point(3, n)
                    cbar = cbar + patch%weight(n)*c(k, j, i)
                END DO
                CALL robin_wall(cwall, jout, conductance, cbar, &
                    patch%ell, diffusivity, sca%geometries(body)%k_s, &
                    sca%geometries(body)%c_eq)
                ! Screen stiffness on every stage, including adaptive dt.
                ! This is a row-norm bound, not a positivity guarantee for
                ! this nonlocal stencil or for centered advection.
                bound = diffusivity*grids(m)%diffusion_rate &
                    + patch%area/patch%volume*conductance
                IF (dt*rkfactor*bound > 0.5_realk) THEN
                    WRITE(*, *) "Robin explicit dt too large; use dt <=", &
                        0.5_realk/(rkfactor*bound)
                    CALL errr(__FILE__, __LINE__)
                END IF
                k = patch%owner(1)
                j = patch%owner(2)
                i = patch%owner(3)
                rate(k, j, i) = patch%area/patch%volume*jout
                rhs(k, j, i) = rhs(k, j, i) - rate(k, j, i)
            END DO
        END DO
    END SUBROUTINE apply_robin


    SUBROUTINE finish_robin()
        IF (ALLOCATED(grids)) DEALLOCATE(grids)
    END SUBROUTINE finish_robin
END MODULE gc_robin_mod
