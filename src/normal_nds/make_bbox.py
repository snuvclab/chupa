import argparse
from distutils.log import error
import os
import numpy as np
from PIL import Image

error_bound = 1e-10 # error bound for computational instability in numpy

cross = lambda x,y:np.cross(x.T,y.T).T # This is for bug resolution in numpy with pylance (related with type checking)
                                       # input : two n × 1 vector
                                       # output : n × 1 vector

def compute_ray_direction(R, K, h, w):
    # output : 3 × 1 vector
    raydir = R.T @ np.linalg.inv(K) @ np.array([w, h, 1]).reshape(-1,1)
    if np.sum(np.abs(raydir)) <= 3 * error_bound:
        # ray directions must not be zero vector
        raise Exception("ray direction cannot be zero. Something wrong")
    return raydir

def dot_product_unit(a, b):
    # input : two n × 1 vector
    # output : 1 × 1 vector
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= error_bound or norm_b <= error_bound:
        return np.zeros((1,1))
    return np.dot((a/norm_a).T,(b/norm_b))

def check_parallel(r1, r2):
    # input : two n × 1 vector
    # output : r1 // r2 ?
    # Assumption(general) : The zero vector is parallel to all vectors
    n = r1.shape[0]
    ratio = None

    # First, check zero vectors
    if np.sum(np.abs(r1)) <= n * error_bound or np.sum(np.abs(r2)) <= n * error_bound:
        return True

    # Next, consider non-zero r1
    for i in range(n):
        if np.abs(r1[i]) <= error_bound:
            # if r1 // r2, r1[i] = 0 => r2[i] = 0 
            if np.abs(r2[i]) > error_bound:
                return False
        else:
            if ratio is None:
                ratio = r2[i] / r1[i]
            else:
                if np.abs(ratio - (r2[i] / r1[i])) > error_bound:
                    return False
    return True

def get_point_of_intersection_between_two_lines(c1, r1, c2, r2):
    # Let consider c1 + λ1 * r1 = c2 + λ2 * r2
    # output : λ1, λ2

    #   c1 + λ1 * r1 = c2 + λ2 * r2
    #   <=> λ1 * r1 = c2 - c1 + λ2 * r2
    #       λ2 * r2 = c1 - c2 + λ1 * r1
    #   <=> λ1 * (r1 × r2) = (c2 - c1) × r2
    #       λ2 * (r2 × r1) = (c1 - c2) × r1
    #   <=> λ1 = dot((c2 - c1) × r2, (r1 × r2)) / ||(r1 × r2)||^2
    #       λ2 = dot((c1 - c2) × r1, (r2 × r1)) / ||(r1 × r2)||^2
    # If two lines are skew, c1 + λ1 * r1 and c2 + λ2 * r2 each represent the closest point to the other line
    cross_norm = np.linalg.norm(cross(r1, r2))
    if cross_norm <= error_bound:
        # It means two lines are parallel or same
        return None, None
    lam1 = np.dot(cross(c2 - c1, r2).T, cross(r1, r2)) / np.square(cross_norm)
    lam2 = np.dot(cross(c1 - c2, r1).T, cross(r2, r1)) / np.square(cross_norm)
    return lam1, lam2

def make_infinite_point_of_ray(r):
    # input : n × 1 vector (direction)
    # output : n × 1 vector (point)
    inf_point = np.zeros(r.shape)
    r_sign = np.sign(r)
    r_inf = np.full(r.shape, np.inf)
    n = r.shape[0]
    
    for i in range(n):
        if np.abs(r[i]) <= error_bound:
            inf_point[i] = 0.0
        else:
            inf_point[i] = r_sign[i] * r_inf[i]
    return inf_point

def get_intersection_line_of_two_planes(c_b, n_b, c_t, n_t):
    # input : centers and normal vectors
    # output : starting point p, direction r

    # Check if two planes are same or parallel
    if check_parallel(n_b, n_t):
        return None, None
    
    # Now, the intersection line must exist
    # This line is parallel with n_t × n_b
    r = cross(n_t, n_b)
    
    # Then, we should get starting point p
    # p is on both planes
    #   dot(n_t, p - c_t) = 0, dot(n_b, p - c_b) = 0 => dot(n_t, p) = dot(n_t, c_t), dot(n_b, p) = dot(n_b, c_b)
    #   We know that r is not zero vector. Thus, there exits i s.t. ri ≠ 0, i = x, y, z
    #   Assume that rx ≠ 0. Then, A = (n_ty n_tz) is invertible
    #                                 (n_by n_bz)
    #   Thus, (py, pz).T = A^-1 @ [dot(n_t, c_t), dot(n_b, c_b)].T where px = 0
    n = r.shape[0]
    p = np.zeros((n,1))
    for i in range(n):
        if np.abs(r[i]) <= error_bound:
            continue
        p[i] = 0
        A = np.vstack([np.vstack([n_t[0:i], n_t[i+1:]]).T, np.vstack([n_b[0:i], n_b[i+1:]]).T])
        B = np.array([np.dot(n_t.T, c_t), np.dot(n_b.T, c_b)]).reshape(2,1)
        p[(i+1)%3], p[(i+2)%3] = np.linalg.inv(A) @ B
        break
    
    return p, r

def get_intersection_of_two_planes(raydirs_base, raydirs_t, c_base, c_t):
    r_b1, r_b2 = raydirs_base
    r_t1, r_t2 = raydirs_t
    c_b = c_base
    intersection_list = []

    # We compute intersection of two planes(one of the base planes and one of the target planes) in this function
    # If we consider infinite points, we can represent intersection of two planes as at most two points
    # First, we can get the equation of two planes by using each center and ray directions
    #   dot(n_t, (x,y,z) - c_t) = 0, dot(n_b, (x,y,z) - c_b) = 0 where n_t = r_t1 × r_t2, n_b = r_b1 × r_b2
    n_t = cross(r_t1, r_t2)
    n_b = cross(r_b1, r_b2)

    # Next, we obtain the intersection of two planes taking into account the relationship between the two planes
    #   1) n_t not // n_b
    #   2) n_t // n_b
    #       2-1) base plane = target plane
    #       2-2) base plane ≠ target plane
    if check_parallel(n_t, n_b):
        # 2)
        # Now, check whether c_t is on base plane
        if np.abs(np.dot(n_b.T, c_t - c_b)) <= error_bound:
            # 2-1)
            # Now we consider follows
            #   2-1-1) Check if each center is in both planes
            #   2-1-2) Check if each ray is in both planes
            #   2-1-3) Obtain intersection of each two lines
            #       i) Check if two lines are parallel
            #       ii) Check if λ >= 0
            #   We only consider 2-1-3) now because the others are considered in `Is_point_in_ray_view` or `Is_ray_in_ray_view`
            for r_b in raydirs_base:
                for r_t in raydirs_t:
                    lam1, lam2  = get_point_of_intersection_between_two_lines(c_b, r_b, c_t, r_t)
                    # check computational stability
                    if (lam1 is not None) and (lam1 >= -error_bound) and (lam2 is not None) and (lam2 >= -error_bound):
                        inter_point_b = c_b + lam1 * r_b
                        inter_point_t = c_t + lam2 * r_t
                        if np.linalg.norm(inter_point_b - inter_point_t) <= error_bound:
                            intersection_list.append(inter_point_b.copy())
                        #else:
                        #    raise Exception("computational stability is not good")
        else:
            # 2-2)
            # There is no intersection
            return intersection_list
    else:
        # 1)
        # Now, let consider intersection line of the two planes(It must exist)
        p, r = get_intersection_line_of_two_planes(c_b, n_b, c_t, n_t) # p, r are not None

        # We check the following for each plane
        #   i) intersection of above line and two rays
        #   ii) if each infinite point of above line is in the rays' plane? 
        base_intersection = []
        target_intersection = []

        # i)
        for r_b in raydirs_base:
            lam1, lam2 = get_point_of_intersection_between_two_lines(p, r, c_b, r_b)
            # check λ2 >= 0
            # We don't have to consider the case where r and r_b are on the same line. This case has already been considered in `Is_point_in_ray_view` or `Is_ray_in_ray_view`
            if (lam1 is not None) and (lam2 is not None) and (lam2 >= -error_bound):
                # check computational stability
                inter_point = p + lam1 * r
                inter_point_b = c_b + lam2 * r_b
                if len(base_intersection) > 0:
                    if np.linalg.norm(c_base - inter_point) <= error_bound:
                        break
                base_intersection.append(inter_point.copy())
                '''
                if np.linalg.norm(inter_point_b - inter_point) <= error_bound:
                    # if the intersection point is already computed, (i.e. intersection point is camera center) just skip
                    if len(base_intersection) > 0:
                        if np.linalg.norm(c_base - inter_point) <= error_bound:
                            break
                    base_intersection.append(inter_point.copy())
                '''
                #else:
                #    raise Exception("computational stability is not good")
        # ii)
        # Two infinite points : p + λ * r, where λ' = inf or -inf
        # At most one of the two infinite points is on the rays' plane
        # The following conditions should be satisfied
        #   1) cos(r, r1) > 0 or cos(r, r2) > 0 (because of FOV < 180 degree)
        #   2) cos(r1, r2) <= cos(r, r1) and cos(r1, r2) <= cos(r, r2)
        for r in (r, -r):
            r1_r2_cos = dot_product_unit(r_b1, r_b2)
            r_r1_cos = dot_product_unit(r, r_b1)
            r_r2_cos = dot_product_unit(r, r_b2)
            if r_r1_cos > error_bound or r_r2_cos > error_bound:
                if (r1_r2_cos <= r_r1_cos) and (r1_r2_cos <= r_r2_cos):
                    inf_point = make_infinite_point_of_ray(r)
                    base_intersection.append(inf_point.copy())
        
        # The same for the target ray
        for r_t in raydirs_t:
            lam1, lam2 = get_point_of_intersection_between_two_lines(p, r, c_t, r_t)
            # check λ2 >= 0
            # We don't have to consider the case where r and r_t are on the same line. This case has already been considered in `Is_point_in_ray_view` or `Is_ray_in_ray_view`
            if (lam1 is not None) and (lam2 is not None) and (lam2 >= -error_bound):
                # check computational stability
                inter_point = p + lam1 * r
                inter_point_t = c_t + lam2 * r_t
                if len(target_intersection) > 0:
                    if np.linalg.norm(c_t - inter_point) <= error_bound:
                        break
                target_intersection.append(inter_point.copy())
                '''
                if np.linalg.norm(inter_point_t - inter_point) <= error_bound:
                    # if the intersection point is already computed, (i.e. intersection point is camera center) just skip
                    if len(target_intersection) > 0:
                        if np.linalg.norm(c_t - inter_point) <= error_bound:
                            break
                    target_intersection.append(inter_point.copy())
                '''
                #else:
                #    raise Exception("computational stability is not good")

        for r in (r, -r):
            r1_r2_cos = dot_product_unit(r_t1, r_t2)
            r_r1_cos = dot_product_unit(r, r_t1)
            r_r2_cos = dot_product_unit(r, r_t2)
            if r_r1_cos > error_bound or r_r2_cos > error_bound:
                if (r1_r2_cos <= r_r1_cos) and (r1_r2_cos <= r_r2_cos):
                    inf_point = make_infinite_point_of_ray(r)
                    target_intersection.append(inf_point.copy())

        # Now, each list has a maximum of 2 points
        assert len(base_intersection) <= 2 and len(target_intersection) <= 2
        
        # Finally, we obtain the intersection of base_intersection and target_intersection
        if len(base_intersection) == 0 or len(target_intersection) == 0:
            # There is no intersection
            return intersection_list

        base_intersection = np.hstack(base_intersection)
        target_intersection = np.hstack(target_intersection)
        # Sort ascending by column
        base_intersection = base_intersection[:, base_intersection[0].argsort()]
        target_intersection = target_intersection[:, target_intersection[0].argsort()]

        # .. -- or . -- or .. -  or . - 
        # Below conditions can cover infinite points 
        if target_intersection[0][0] > base_intersection[0][-1] \
            or base_intersection[0][0] > target_intersection[0][-1]:
            # There is no intersection
            return intersection_list

        # Now consider intersection is a point
        # -.- or .-. or . == -
        if base_intersection.shape[1] == 1:
            intersection_list.append(base_intersection[:, 0].reshape(3, 1))
            return intersection_list
        if target_intersection.shape[1] == 1:
            intersection_list.append(target_intersection[:, 0].reshape(3, 1))
            return intersection_list
        
        # Finally, consider intersectoin is a line segment or ray
        #   len(base_intersection) == 2 and len(target_intersection) == 2
        # -.-. or .--. or -..- or .-.-
        all_intersection = np.hstack([base_intersection, target_intersection])
        all_intersection = all_intersection[:, all_intersection[0].argsort()]
        intersection_list.append(all_intersection[:, 1].reshape(3,1))
        intersection_list.append(all_intersection[:, 2].reshape(3,1))
    
    return intersection_list

def get_intersection_of_rays_and_bbox(bbox, raydirs, c_t):
    # bbox = xmax, ymax, zmax, xmin, ymin, zmin
    # raydirs = r1, r2
    intersection_list = []
    r1, r2 = raydirs

    # We can cut the 3D bbox using intersection of the masks' rays and the 3D bbox
    # It is performed according to the following algorithm:
    #   1) Find the intersection of the two rays and the bbox's faces
    #       1-1) If each ray is perpendicular to the normal vector of plane, there is no intersection
    #            (i.e. That is, intersection points are not considered when each ray is on the plane as well as when it is parallel to the plane)
    #       1-2) check λ >= 0 (In fact, the bbox is always in front of all cameras. So we don't need to check it. But.. we check it just for safety)
    #   2) Find the intersection of the rays' plane and bbox's edges

    # 1)
    for r in raydirs:
        for i in range(r.shape[0]):
            if np.abs(r[i]) <= error_bound:
                # The ray is perpendicular to the i_axis(this is the normal vector of planes which are i = i_max, i = i_min), i = x, y, z
                continue
            else:
                # get λ which satisfies c_t[i] + λ * r[i] = i_max, i_min
                i_max, i_min = bbox[i], bbox[i+3]
                lam_max = (i_max - c_t[i]) / r[i]
                lam_min = (i_min - c_t[i]) / r[i]
                if lam_max >= -error_bound:
                    inter_point = c_t + lam_max * r
                    if (bbox[((i+1)%3)+3] <= inter_point[(i+1)%3] and inter_point[(i+1)%3] <= bbox[(i+1)%3]) and \
                        (bbox[((i+2)%3)+3] <= inter_point[(i+2)%3] and inter_point[(i+2)%3] <= bbox[(i+2)%3]):
                        intersection_list.append(inter_point.copy())
                if lam_min >= -error_bound:
                    inter_point = c_t + lam_min * r
                    if (bbox[((i+1)%3)+3] <= inter_point[(i+1)%3] and inter_point[(i+1)%3] <= bbox[(i+1)%3]) and \
                        (bbox[((i+2)%3)+3] <= inter_point[(i+2)%3] and inter_point[(i+2)%3] <= bbox[(i+2)%3]):
                        intersection_list.append(inter_point.copy())
    # 2)
    # First, consider the rays' plane which constructed by c_t, r1, r2 : dot(n, (x,y,z) - c_t) = 0, n = r1 × r2
    # Next, consider edges of bbox
    #   There are 12 edges
    #   Each edge has a fixed value for two axes, and bounded values(end points of interval) for the other axis
    # Now, consider an edge with y and z values y0 and z0, and x values from x_min to x_max
    #   1) dot(n, (x,y0,z0) - c_t) = 0 => n_x(x - c_tx) = -n_y(y0 - c_ty) - n_z(z0 - c_tz)
    #   2) If n_x = 0, the edge is on the plane or the edge and the plane are parallel. This does not need to be considered because 
    #       2-1) If the edge and rays don't meet, we don't need to cut the bbox
    #       2-2) If the edge and rays meet, this case was considered in 1) already
    #   3) If n_x ≠ 0, the edge and the plane meet at most one point : x0 = (-n_y(y0 - c_ty) - n_z(z0 - c_tz)) / n_x + c_tx
    #   4) Check the point is valid
    #       4-1) x_min <= x0 <= x_max ?
    #       4-2) Is the point in between the two rays?
    n = cross(r1, r2)
    for i in range(n.shape[0]):
        if np.abs(n[i]) <= error_bound:
            continue
        i_min, i_max = bbox[i+3], bbox[i]
        fixed_value_1 = bbox[((i+1)%3) + 3], bbox[(i+1)%3]
        fixed_value_2 = bbox[((i+2)%3) + 3], bbox[(i+2)%3]
        for j in range(2):
            v1 = fixed_value_1[j]
            for k in range(2):
                v2 = fixed_value_2[k]
                # get point of intersection
                inter_point = np.zeros((3,1))
                inter_point[i] = (-n[(i+1)%3] * (v1 - c_t[(i+1)%3]) -n[(i+2)%3] * (v2 - c_t[(i+2)%3])) / n[i] + c_t[i]
                inter_point[(i+1)%3] = v1
                inter_point[(i+2)%3] = v2
                # check validation 4-1)
                if i_min <= inter_point[i] and inter_point[i] <= i_max:
                    # check validation 4-2)
                    #   1) cos(r, r1) > 0 or cos(r, r2) > 0 (because of FOV < 180 degree)
                    #   2) cos(r1, r2) <= cos(r, r1) and cos(r1, r2) <= cos(r, r2)
                    r = inter_point - c_t
                    r1_r2_cos = dot_product_unit(r1, r2)
                    r_r1_cos = dot_product_unit(r, r1)
                    r_r2_cos = dot_product_unit(r, r2)
                    if r_r1_cos > error_bound or r_r2_cos > error_bound:
                        if (r1_r2_cos <= r_r1_cos) and (r1_r2_cos <= r_r2_cos):
                            # point of intersection exists on ray plane!
                            intersection_list.append(inter_point.copy())
                
    return intersection_list

def Is_point_in_ray_view(p, ray_pair_list, c):
    # ray_pair_list = [(r_top_left, r_top_right), ...]
    
    # All cross products of two rays (i.e. r1 × r2) have inward direction of the ray's field of view. Because our ray order is set clock wise manner
    # So, we do the following:
    #   1) get the normal vectors from each of the two rays
    #   2) get the normal vectors from the point to each plane
    #   3) The directions of each normal vector pair in 1) and 2) must be opposite
    #   Let denote n  = r1 × r2, and camera center = c, a point = p
    #   Then we can find the normal vector at point p by solving the following equation
    #       dot(n, p + λ*n - c) = 0 => λ = dot(n, c - p) / ||n||^2
    #   So the normal vector at point p is λ*n (= p + λ*n - p)
    #   If λ <= 0 for all plane, p is in rays' field of view
    #       Since ||n|| ≠ 0, λ <= 0 <=> dot(n, c - p) <= 0

    # 1)
    ray_normal_list = []
    for ray_pair in ray_pair_list:
        ray_normal_list.append(cross(ray_pair[0], ray_pair[1]))
    
    # 2), 3)
    is_in_ray = True
    for n in ray_normal_list:
        # norm of normal cannot be zero
        if np.linalg.norm(n) <= error_bound:
            raise Exception("normal cannot be zero vector")
        if np.dot(n.T, c - p) > error_bound:
            is_in_ray = False
            break 
    return is_in_ray

def Is_ray_in_ray_view(r, c_t, ray_pair_list, c):
    # ray_pair_list = [(r_top_left, r_top_right), ...]
    
    # Is the target ray r in the base rays's field of view?
    # We just need to check the point, p = c_t + λ' * r, is in the base rays's field of view when λ' = inf
    # All cross products of two rays (i.e. r1 × r2) have inward direction of the ray's field of view. Because our ray order is set clock wise manner
    # So, we do the following:
    #   1) get the normal vectors from each of the two rays
    #   2) get the normal vectors from the point to each plane
    #   3) The directions of each normal vector pair in 1) and 2) must be opposite
    #   Let denote n  = r1 × r2, and camera center = c, a point = p = c_t + λ' * r
    #   Then we can find the normal vector at point p by solving the following equation
    #       dot(n, p + λ*n - c) = 0 => λ = dot(n, c - p) / ||n||^2
    #   So the normal vector at point p is λ*n (= p + λ*n - p)
    #   If λ <= 0 for all plane, p is in rays' field of view
    
    #   Now, let's simplify the above process(λ' = inf)
    #   λ <= 0 <=> dot(n, c - p) <= 0 <=> dot(n, c - c_t - λ' * r) <= 0 <=> dot(n, c - c_t) - λ' * dot(n, r) <= 0 <=> [dot(n, r) > 0] or [dot(n, r) = 0, dot(n, c - c_t) <= 0] 

    # 1)
    ray_normal_list = []
    for ray_pair in ray_pair_list:
        ray_normal_list.append(cross(ray_pair[0], ray_pair[1]))
    
    # 2), 3)
    is_in_ray = True
    for n in ray_normal_list:
        # norm of normal cannot be zero
        if np.linalg.norm(n) <= error_bound:
            raise Exception("normal cannot be zero vector")
        if np.dot(n.T, r) < -error_bound:
            is_in_ray = False
            break
        elif np.abs(np.dot(n.T, r)) <= error_bound:
            if np.dot(n.T, c - c_t) > error_bound:
                is_in_ray = False
                break
    return is_in_ray

def get_bbox_corners_in_ray_view(bbox, ray_pair_list, c_t):
    # bbox = xmax, ymax, zmax, xmin, ymin, zmin
    # ray_pair_list = [(r_top_left, r_top_right), ...]
    
    # All cross products of two rays (i.e. r1 × r2) have inward direction of the ray's field of view. Because our ray order is set clock wise manner
    # So, we do the following:
    #   1) get the normal vectors from each of the two rays
    #   2) get the normal vectors from the corner of bbox to each plane
    #   3) The directions of each normal vector pair in 1) and 2) must be opposite
    #   Let denote n  = r1 × r2, and camera center = c_t, a corner point = p
    #   Then we can find the normal vector at point p by solving the following equation
    #       dot(n, p + λ*n - c_t) = 0 => λ = dot(n, c_t - p) / ||n||^2
    #   So the normal vector at point p is λ*n (= p + λ*n - p)
    #   If λ <= 0 for all plane, p is in rays' field of view

    # 1)
    ray_normal_list = []
    for ray_pair in ray_pair_list:
        ray_normal_list.append(cross(ray_pair[0], ray_pair[1]))
    
    # 2), 3)
    corner_in_rays_list = []
    corner = np.zeros((3,1))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                corner[0] = bbox[i*3]
                corner[1] = bbox[j*3 + 1]
                corner[2] = bbox[k*3 + 2]

                is_in_ray = True
                for n in ray_normal_list:
                    # norm of normal cannot be zero
                    if np.linalg.norm(n) <= error_bound:
                        raise Exception("normal cannot be zero vector")
                    lam = np.dot(n.T, c_t - corner) / np.square(np.linalg.norm(n))
                    if lam > error_bound:
                        is_in_ray = False
                        break        
                if is_in_ray:
                    corner_in_rays_list.append(corner.copy())
    
    return corner_in_rays_list

def make_bbox(masks, cam, extend_ratio, reliable_ratio, use_inf_bbox_to_bounded=False):
    '''
    Robust 3D bounding box generation algorithm with calibrated cameras and 2D bounding boxes
    Assumptions
        1) A few 2D bounding boxes (masks) are poor, but most masks are reliable
            1-1) 'poor' includes not only the segmentation quality, but also whether the mask completely contains the object
                 Thus, Most images should completely contain the object(i.e. the image quality is also included)
        2) There is an area that all camera views see in common
        3) Rays(half lines) goes in the positive direction : ray = c + λ * r, λ >= 0
        4) FOV is less than 180 degree for all cameras
        5) Camera format should be OpenCV : X_im = K (R @ X_world + t)
        6) image coordinate : x - right, y - bottom
    All 3D points and direction vectors have (3,1) shape
    Time complexity
        1) When all views(images) completely contain the object(so, object is not clipped)
            O(n), n = # of views(masks)
        2) Some views do not fully contain objects(so, object is clipped in some images)
            O(n^2)
        ----------------------------------------------------------------------
        Above versions are not implemented because they can be covered some extent by heuristics : just using `extend_ratio`
        (When the ratio of the cut part is small compared to the overall size of the object)
        3) We assume that objects in some views are partially clipped
            O(n)
    '''

    base_mask = list(masks.keys())[0]
    select_ray = [("top_left","top_right"), ("top_right","bottom_right"), ("bottom_right","bottom_left"), ("bottom_left","top_left")] # clock-wise manner for right-had rule
    
    # compute camera center and ray direction(ray direction for image corner)
    # Let c : camera center, r : ray direction, then equation of ray : c + λ * r, λ >= 0
    cam_center_dict = {}
    ray_direction_dict = {}
    for mask_name in masks.keys():
        frame_num = cam[mask_name]
        pose = cam["pose_"+str(frame_num)]
        R, t = pose[:3, :3], pose[:3, 3:]
        K = cam["intrinsic_"+str(frame_num)]
        cam_center_dict[mask_name] = -R.T @ t
        # image coordinate : x - right, y - bottom
        y, x = np.where(masks[mask_name])
        ray_direction_dict[mask_name] = {"top_left": compute_ray_direction(R, K, y.min(), x.min()), "top_right": compute_ray_direction(R, K, y.min(), x.max()),
                                    "bottom_left": compute_ray_direction(R, K, y.max(), x.min()), "bottom_right": compute_ray_direction(R, K, y.max(), x.max())}
    
    # Check validation
    # Assumption
    #   1. Rays goes in the positive direction. ray = c + λ * r, λ >= 0
    #   2. FOV is less than 180 degree for all cameras
    #   3. Ignore distortion of cameras
    # We assume 3 then validate 1, 2.
    # How?
    #   R.T @ (0,0,1).T : camera center ray ; X_cam = R @ X_world + t => R.T @ λ*(0, 0, 1).T - R.T @ t = X_world => camera center ray = R.T @ (0, 0, 1).T, λ >= 0
    #   If dot((unit(camera center ray), unit(r)) <= 0, then 1 or 2 is false (But this can not detect [1 false and 2 false]. We ignore it)
    #   This is not a direct measurement of FOV, but it doesn't matter. This is equivalent
    for mask_name in masks.keys():
        frame_num = cam[mask_name]
        pose = cam["pose_"+str(frame_num)]
        R = pose[:3, :3]
        cam_center_ray = R.T @ np.array([0, 0, 1]).reshape(3,1)
        for key in ray_direction_dict[mask_name].keys():
            if dot_product_unit(cam_center_ray, ray_direction_dict[mask_name][key]) <= error_bound:
                raise Exception("Camera ray doesn't satisfy assumption!\n\
                    mask_name : {mask_name},    frame_num : {frame_num}".format(mask_name=mask_name, frame_num=cam[mask_name]))
    
    # get 3D bboxes made by two 2D bbox (base view and each of the other views)
    bbox_3d_each_view_dict = {}
    c_base = cam_center_dict[base_mask] # -R0.T @ t0
    base_raydir_dict = ray_direction_dict[base_mask]

    for mask_name in masks.keys():
        if mask_name == base_mask:
            continue
        bbox_3d_each_view_dict[mask_name] = []
        c_t = cam_center_dict[mask_name] # -R.T @ t
        target_raydir_dict = ray_direction_dict[mask_name]

        # Compute intersection of two planes(base plane and target palne)
        for base_order in select_ray:
            r_b1 = base_raydir_dict[base_order[0]]
            r_b2 = base_raydir_dict[base_order[1]]
            for target_order in select_ray:
                r_t1 = target_raydir_dict[target_order[0]]
                r_t2 = target_raydir_dict[target_order[1]]
                bbox_3d_each_view_dict[mask_name].extend(get_intersection_of_two_planes((r_b1, r_b2), (r_t1, r_t2), c_base, c_t))

        # Check the camera center is in the rays's field of view
        base_ray_pairs = []
        target_ray_pairs = []
        for ray_order in select_ray:
            base_r1 = base_raydir_dict[ray_order[0]]
            base_r2 = base_raydir_dict[ray_order[1]]
            base_ray_pairs.append((base_r1, base_r2))
            t_r1 = target_raydir_dict[ray_order[0]]
            t_r2 = target_raydir_dict[ray_order[1]]
            target_ray_pairs.append((t_r1, t_r2))
        if Is_point_in_ray_view(c_t, base_ray_pairs, c_base):
            bbox_3d_each_view_dict[mask_name].append(c_t)
        if Is_point_in_ray_view(c_base, target_ray_pairs, c_t):
            bbox_3d_each_view_dict[mask_name].append(c_base)

        # Check the target(base) ray is in the base(target) rays' field of view
        #   If target(base) ray is in the base(target) rays' field of view, intersection point has infinite value
        for key in target_raydir_dict.keys():
            r = target_raydir_dict[key]
            if Is_ray_in_ray_view(r, c_t, base_ray_pairs, c_base):
                bbox_3d_each_view_dict[mask_name].append(make_infinite_point_of_ray(r))
        for key in base_raydir_dict.keys():
            r = base_raydir_dict[key]
            if Is_ray_in_ray_view(r, c_base, target_ray_pairs, c_t):
                bbox_3d_each_view_dict[mask_name].append(make_infinite_point_of_ray(r))

    # use minmax for 3D bbox
    #   First, 3D bboxes including a convex polyhedron obtained from each target view are constructed
    #   Next, get a 3D bbox, which is the intersection of all above 3D bboxes
    xmax, ymax, zmax = np.inf, np.inf, np.inf
    xmin, ymin, zmin = -np.inf, -np.inf, -np.inf
    for key in bbox_3d_each_view_dict.keys():
        vertex_list = bbox_3d_each_view_dict[key]
        if len(vertex_list) > 0:
            vertices = np.hstack(vertex_list) # 3 × len(vertex_list) vector
            max_points = np.max(vertices, -1)
            min_points = np.min(vertices, -1)
            
            xmax = max_points[0] if xmax > max_points[0] else xmax
            ymax = max_points[1] if ymax > max_points[1] else ymax
            zmax = max_points[2] if zmax > max_points[2] else zmax

            xmin = min_points[0] if xmin < min_points[0] else xmin
            ymin = min_points[1] if ymin < min_points[1] else ymin
            zmin = min_points[2] if zmin < min_points[2] else zmin

            if xmax <= xmin or ymax <= ymin or zmax <= zmin:
                import pdb
                pdb.set_trace()
    
    # Check validatoin
    # Assumption : There is an area that all camera views see in common
    # Thus, we check _min < _max
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise Exception("There should be an area that all camera views see in common")

    # bbox should have bounded size
    # When generating meshes for the train, the bbox should be clear. Therefore, scenes in which bbox has an inf value are excluded for train
    # However, there are some cases when we want to train only the visible part without worrying about the invisible part
    # For such cases, set bbox to force
    bbox = [xmax, ymax, zmax, xmin, ymin, zmin]
    if use_inf_bbox_to_bounded:
        max_value = np.max(np.where(np.abs(bbox) < np.inf, np.abs(bbox), 0))
        for i in range(len(bbox)):
            if np.abs(bbox[i]) == np.inf:
                bbox[i] = max_value * np.sign(bbox[i])
    # Exclude bbox which has infinite value
    else:
        for i in range(len(bbox)):
            if np.abs(bbox[i]) == np.inf:
                return (None, None, None, None, None, None)

    # For clipped object masks
    # reliable_ratio isn't used now
    for i in range(3):
        mid_point = (bbox[i] + bbox[i+3]) / 2.0
        half_dist = (bbox[i] - bbox[i+3]) / 2.0
        bbox[i] = mid_point + extend_ratio * half_dist
        bbox[i+3] = mid_point - extend_ratio * half_dist

    return bbox

    '''
    For 1) and 2)
    base_mask = list(masks.keys())[0]
    select_ray = [("top_left","top_right"), ("top_right","bottom_right"), ("bottom_right","bottom_left"), ("bottom_left","top_left")] # clock-wise manner for right-had rule
    
    # compute camera center and ray direction(ray direction for image corner)
    # Let c : camera center, r : ray direction, then equation of ray : c + λ * r, λ >= 0
    cam_center_dict = {}
    ray_direction_dict = {}
    for mask_name in masks.keys():
        frame_num = cam[mask_name]
        pose = cam["pose_"+str(frame_num)]
        R, t = pose[:3, :3], pose[:3, 3:]
        K = cam["intrinsic_"+str(frame_num)]
        cam_center_dict[mask_name] = -R.T @ t
        # image coordinate : x - right, y - bottom
        H, W = masks[mask_name].shape
        ray_direction_dict[mask_name] = {"top_left": compute_ray_direction(R, K, 0, 0), "top_right": compute_ray_direction(R, K, 0, W),
                                    "bottom_left": compute_ray_direction(R, K, H, 0), "bottom_right": compute_ray_direction(R, K, H, W)}

    # Check validation
    # Assumption
    #   1. Rays goes in the positive direction. ray = c + λ * r, λ >= 0
    #   2. FOV is less than 180 degree for all cameras
    #   3. Ignore distortion of cameras
    # We assume 3 then validate 1, 2.
    # How?
    #   R.T @ t : vector that camera center to origin
    #   If dot((unit(R.T @ t), unit(r)) <= 0, then 1 or 2 is false
    #   This is not a direct measurement of FOV, but it doesn't matter. This is equivalent
    for mask_name in masks.keys():
        center_to_orient_vec = -cam_center_dict[mask_name]
        for key in ray_direction_dict[mask_name].keys():
            if dot_product_unit(center_to_orient_vec, ray_direction_dict[mask_name][key]) <= error_bound:
                raise Exception("Camera ray doesn't satisfy assumption!\n\
                    mask_name : {mask_name},    frame_num : {frame_num}".format(mask_name=mask_name, frame_num=cam[mask_name]))

    ############################################# First, get 3D boudning box from image corners #############################################
    #                                                       Why perform this process?                                                       #
    #                          we can't know which mask is reliable(in automation pipeline). So we can't set base mask                      #
    #########################################################################################################################################
    
    # get 3D bboxes made by two views (base view and each of the other views)
    bbox_3d_each_view_dict = {}
    c_base = cam_center_dict[base_mask] # -R0.T @ t0
    base_raydir_dict = ray_direction_dict[base_mask]

    for mask_name in masks.keys():
        if mask_name == base_mask:
            continue
        bbox_3d_each_view_dict[mask_name] = []
        c_t = cam_center_dict[mask_name] # -R.T @ t
        target_raydir_dict = ray_direction_dict[mask_name]
        # Compute intersection of two planes(base plane and target palne)
        for base_order in select_ray:
            r_b1 = base_raydir_dict[base_order[0]]
            r_b2 = base_raydir_dict[base_order[1]]
            for target_order in select_ray:
                r_t1 = target_raydir_dict[target_order[0]]
                r_t2 = target_raydir_dict[target_order[1]]
                bbox_3d_each_view_dict[mask_name].extend(get_intersection_of_two_planes((r_b1, r_b2), (r_t1, r_t2), c_base, c_t))

        # Check the camera center is in the rays's field of view
        base_ray_pairs = []
        target_ray_pairs = []
        for ray_order in select_ray:
            base_r1 = base_raydir_dict[ray_order[0]]
            base_r2 = base_raydir_dict[ray_order[1]]
            base_ray_pairs.append((base_r1, base_r2))
            t_r1 = target_raydir_dict[ray_order[0]]
            t_r2 = target_raydir_dict[ray_order[1]]
            target_ray_pairs.append((t_r1, t_r2))
        if Is_point_in_ray_view(c_t, base_ray_pairs, c_base):
            bbox_3d_each_view_dict[mask_name].append(c_t)
        if Is_point_in_ray_view(c_base, target_ray_pairs, c_t):
            bbox_3d_each_view_dict[mask_name].append(c_base)

        # Check the target(base) ray is in the base(target) rays' field of view
        #   If target(base) ray is in the base(target) rays' field of view, intersection point has infinite value
        for key in target_raydir_dict.keys():
            r = target_raydir_dict[key]
            if Is_ray_in_ray_view(r, c_t, base_ray_pairs, c_base):
                bbox_3d_each_view_dict[mask_name].append(make_infinite_point_of_ray(r))
        for key in base_raydir_dict.keys():
            r = base_raydir_dict[key]
            if Is_ray_in_ray_view(r, c_base, target_ray_pairs, c_t):
                bbox_3d_each_view_dict[mask_name].append(make_infinite_point_of_ray(r))

    # use minmax for 3D bbox
    #   First, 3D bboxes including a convex polyhedron obtained from each target view are constructed
    #   Next, get a 3D bbox, which is the intersection of all above 3D bboxes
    xmax, ymax, zmax = np.inf, np.inf, np.inf
    xmin, ymin, zmin = -np.inf, -np.inf, -np.inf
    for key in bbox_3d_each_view_dict.keys():
        vertex_list = bbox_3d_each_view_dict[key]
        if len(vertex_list) > 0:
            vertices = np.hstack(vertex_list) # 3 × len(vertex_list) vector
            max_points = np.max(vertices, -1)
            min_points = np.min(vertices, -1)
            
            xmax = max_points[0] if xmax > max_points[0] else xmax
            ymax = max_points[1] if ymax > max_points[1] else ymax
            zmax = max_points[2] if zmax > max_points[2] else zmax

            xmin = min_points[0] if xmin < min_points[0] else xmin
            ymin = min_points[1] if ymin < min_points[1] else ymin
            zmin = min_points[2] if zmin < min_points[2] else zmin
    
    # Check validatoin
    # Assumption : There is an area that all camera views see in common
    # Thus, we check _min < _max
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise Exception("There should be an area that all camera views see in common")
    
    # bbox should have bounded size
    # When generating meshes for the train, the bbox should be clear. Therefore, scenes in which bbox has an inf value are excluded for train
    # However, there are some cases when we want to train only the visible part without worrying about the invisible part
    # For such cases, set bbox to force
    if use_inf_bbox_to_bounded:
        max_value = np.max(np.where(np.abs(bbox) < np.inf, np.abs(bbox), 0))
        for i in range(len(bbox)):
            if np.abs(bbox[i]) == np.inf:
                bbox[i] = max_value * np.sign(bbox[i])
    # Exclude bbox which has infinite value
    else:
        for i in range(len(bbox)):
            if np.abs(bbox[i]) == np.inf:
                return (None, None, None, None, None, None)

    ########################################### Next, modify 3D boudning box from 2D bbox corners ###########################################
    #                                          Cut 3D bbox which is obtaied above using 2D bboxes                                           # 
    #########################################################################################################################################

    bbox_3d_each_mask_dict = {}
    for mask_name in masks.keys():
        bbox_3d_each_mask_dict[mask_name] = []
        H, W = masks[mask_name].shape

        # get 2D bbox
        y, x = np.where(masks[mask_name])
        # get camera center
        c_t = cam_center_dict[mask_name]
        # get camera matrix
        frame_num = cam[mask_name]
        pose = cam["pose_"+str(frame_num)]
        R = pose[:3, :3]
        K = cam["intrinsic_"+str(frame_num)]
        # get rays from 2D bbox corners
        # If an object is clipped from the image, the ray that cuts the object should not cut the bbox 
        rays = {}
        if x.min() > 0 and y.min() > 0:
            rays["top_left"] = compute_ray_direction(R, K, y.min(), x.min())
        if x.max() < W - 1 and y.min() > 0:
            rays["top_right"] = compute_ray_direction(R, K, y.min(), x.max())
        if x.max() < W - 1 and y.max() < H - 1:
            rays["bottom_right"] = compute_ray_direction(R, K, y.max(), x.max())
        if x.min() > 0 and y.max() < H - 1:
            rays["bottom_left"] = compute_ray_direction(R, K, y.max(), x.min())

        valid_ray_pairs = []
        for ray_order in select_ray:
            if ray_order[0] in rays.keys() and ray_order[1] in rays.keys():
                r1 = rays[ray_order[0]]
                r2 = rays[ray_order[1]]
                valid_ray_pairs.append((r1, r2))

        for ray_pair in valid_ray_pairs:
            # get intersection of rays and bbox
            bbox_3d_each_mask_dict[mask_name].extend(get_intersection_of_rays_and_bbox(bbox, ray_pair, c_t))
        
        # To make it clear what will be cut, we add the corners of the bbox within the rays's field of view
        # Use right-hand rule for cross product
        if len(valid_ray_pairs) > 0:
            bbox_3d_each_mask_dict[mask_name].extend(get_bbox_corners_in_ray_view(bbox, valid_ray_pairs, c_t))

    # use robust minmax for 3D bbox
    #   First, 3D bboxes including a convex polyhedron obtained from each target mask are constructed
    #   Next, get a 3D bbox, which is the intersection of all the above `reliable` 3D bboxes
    #       Let denote r := reliable_ratio, then we ignore the max values of the lower 1-r ratios and the min values of the upper 1-r ratios.
    max_points = None
    min_points = None
    for key in bbox_3d_each_mask_dict.keys():
        vertex_list = bbox_3d_each_mask_dict[key]
        if len(vertex_list) > 0:
            vertices = np.hstack(vertex_list) # 3 × len(vertex_list) vector
            if max_points is None:
                max_points = np.max(vertices, -1)
            else:
                max_points = np.vstack([max_points, np.max(vertices, -1)])
            if min_points is None:
                min_points = np.min(vertices, -1)
            else:
                min_points = np.vstack([min_points, np.min(vertices, -1)])

    max_points = np.sort(max_points, 0)
    min_points = np.sort(min_points, 0)[::-1]

    N = len(masks)
    select_num = N - int(N*reliable_ratio)
    if select_num == N:
        raise Exception("There is no appropriate 3D bbox")
    xmax, ymax, zmax = max_points[select_num]
    xmin, ymin, zmin = min_points[select_num]
    
    # Check validatoin
    if xmax <= xmin or ymax <= ymin or zmax <= zmin:
        raise Exception("Most of masks should be reliable")
    
    return xmax, ymax, zmax, xmin, ymin, zmin
    '''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", required=True, type=str)
    parser.add_argument("--camera_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--cam_file_type", required=True, type=str, help="txt | npz")
    parser.add_argument("--mask_threshold", default=0.3, type=float, help="a parameter that changes the mask values to binary when mask values are continuous")
    parser.add_argument("--reliable_ratio", default=1.0, type=float, help="rate of reliable masks")
    parser.add_argument("--extend_ratio", default=1.2, type=float, help="1 / extend_ratio = clipped ratio")
    parser.add_argument("--use_inf_bbox_to_bounded", action="store_true", help="Bound the box with the inf value to an appropriate value")
    args = parser.parse_args()

    # get mask and camera
    masks_list = os.listdir(args.mask_dir)
    mask_paths = [os.path.join(args.mask_dir, mask) for mask in masks_list if (".png" in mask or ".jpg" in mask)]

    cam_file_type = args.cam_file_type
    cam = {}
    if cam_file_type == "txt":
        # txt type : assume camera parameter in txt file and file names of img and cam are same
        for i, mask in enumerate(mask_paths):
            mask_name = mask.split("/")[-1]
            k_path = os.path.join(args.mask_dir, mask_name.split(".")[0]+"_k.txt")
            r_path = os.path.join(args.mask_dir, mask_name.split(".")[0]+"_r.txt")
            t_path = os.path.join(args.mask_dir, mask_name.split(".")[0]+"_t.txt")
            if os.path.isfile(k_path) and os.path.isfile(r_path) and os.path.isfile(t_path):
                K = np.loadtxt(os.path.join(args.mask_dir, mask_name.split(".")[0]+"_k.txt"))
                pose = np.eye(4)
                pose[:3, :3] = np.loadtxt(os.path.join(args.mask_dir, mask_name.split(".")[0]+"_r.txt"))
                pose[:3, 3] = np.loadtxt(os.path.join(args.mask_dir, mask_name.split(".")[0]+"_t.txt"))       
                cam[mask_name] = str(i)
                cam["pose_"+str(i)] = pose
                cam["intrinsic_"+str(i)] = K
    elif cam_file_type == "npz":
        # npz type : assume there is just one npz file in the camera directory
        # assume cam has (image_file_name, frame num) pair
        cam_org_path = [path for path in os.listdir(args.camera_dir) if ".npz" in path]
        if len(cam_org_path) != 1:
            raise Exception("There should be only one camera file!")
        cam_org_path = cam_org_path[0]
        cam_org = np.load(os.path.join(args.camera_dir, cam_org_path))
        for mask_path in mask_paths:
            mask_name = mask_path.split("/")[-1]
            image_name = ".".join(mask_name.split(".")[0:2])
            if image_name in cam_org.keys():
                frame_num = str(cam_org[image_name])
                cam[mask_name] = frame_num
                cam["pose_"+frame_num] = cam_org["pose_"+frame_num]
                cam["intrinsic_"+frame_num] = cam_org["intrinsic_"+frame_num]    
    else:
        raise Exception("data type must dvr or co3d")
    
    masks = {}
    for mask_path in mask_paths:
        mask_name = mask_path.split("/")[-1]
        if cam_file_type == "txt":
            if mask_name not in cam.keys():
                continue
        elif cam_file_type == "npz":
            if mask_name not in cam.keys():
                continue
        else:
            raise Exception("data type must dvr or co3d")

        mask = np.array(Image.open(mask_path), dtype=np.float32)
        mask /= 255.0
        if len(mask.shape) == 3:
            # Does the alpha value role as a mask?
            if mask.shape[2] == 4:
                mask = mask[:, :, 3]
            else:
                mask = mask[:, :, 0]
        mask = (mask > args.mask_threshold)
        
        # Exclude black image
        y, x = np.where(mask)
        if len(x) != 0 and len(y) != 0:
            masks[mask_name] = mask

    # make 3d bbox
    xmax, ymax, zmax, xmin, ymin, zmin = make_bbox(masks, cam, args.extend_ratio, args.reliable_ratio, args.use_inf_bbox_to_bounded)
    try:
        xmax, ymax, zmax, xmin, ymin, zmin = make_bbox(masks, cam, args.extend_ratio, args.reliable_ratio, args.use_inf_bbox_to_bounded)
    except:
        print("Some error in {mask_dir}".format(mask_dir=args.mask_dir))
        return
    if xmax == None:
        # Infinity bounding box
        print("We can't make bbox for {mask_dir}".format(mask_dir=args.mask_dir))
        return

    bbox_path = os.path.join(args.output_dir, "bbox.txt")
    with open(bbox_path, 'w') as f:
        f.write("{xmin} {ymin} {zmin}\n".format(xmin=xmin, ymin=ymin, zmin=zmin))
        f.write("{xmax} {ymax} {zmax}".format(xmax=xmax, ymax=ymax, zmax=zmax))

    return

if __name__ == "__main__":
    main()